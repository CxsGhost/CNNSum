import os 
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import gc
import json
import jsonlines
import sys
import os
import jieba
import numpy as np
from tqdm import tqdm
from rouge import Rouge
from transformers import AutoConfig
from vllm import LLM, SamplingParams
import torch
from absl import app
from absl import flags
import logging
sys.setrecursionlimit(10000)


logger = logging.getLogger('inference_log')
logger.setLevel(logging.INFO)


FLAGS = flags.FLAGS
flags.DEFINE_string('base_model', None, 'local model path', required=True)
flags.DEFINE_string('output_path', None, 'model outputs save path', required=False)
flags.DEFINE_string('data_path', None, 'local data path', required=False)
flags.DEFINE_bool('enable_extrapolate', True, 'allow model extrapolate', required=False)
flags.DEFINE_list('subsets', ['L', 'XL', '2XL', '3XL'], 'subsets to use', required=False)
flags.DEFINE_float('gpu_memory_utilization', 0.97, 'vllm gpu_memory_utilization', required=False)
flags.DEFINE_float('temperature', 0.0, 'vllm sampling params', required=False)
flags.DEFINE_float('top_p', 1.0, 'vllm sampling params', required=False)
flags.DEFINE_integer('top_k', -1, 'vllm sampling params', required=False)
flags.DEFINE_integer('tensor_parallel_size', 1, 'vllm tensor_parallel_size', required=False)
flags.DEFINE_integer('vllm_model_len', None, 'vllm model context length', required=False)
flags.DEFINE_integer('cpu_offload_gb', 0, 'vllm cpu_offload_gb', required=False)


# ref CLongEval
def rouge_zh_score(prediction, ground_truth):
    if prediction == "":
        return 0.
    if set(list(prediction)) == {'.'}:
        return 0.
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    rouge = Rouge()
    scores_one_sample = rouge.get_scores(prediction, ground_truth)
    return scores_one_sample[0]["rouge-l"]["f"]


# ref CLongEval
def scorer(predictions, answers):
    assert len(predictions) == len(answers)
    final_scores = []
    for prediction, ground_truth in zip(predictions, answers):
        score = rouge_zh_score(prediction=prediction, ground_truth=ground_truth)
        final_scores.append(score)
    return round(100 * sum(final_scores) / len(predictions), 5), final_scores
            

def main(args):
    del args
    model_config = AutoConfig.from_pretrained(FLAGS.base_model)
    rope_scaling = {'enable_extrapolate': FLAGS.enable_extrapolate}
    rope_scaling = rope_scaling.update(model_config.get('rope_scaling', {}))
    
    default_infer_len = {'L': 400, 'XL': 400, '2XL': 500, '3XL': 500}
    default_max_len = {'L': 1024 * 32, 
                       'XL': 1024 * 64, 
                       '2XL': 1024 * 128, 
                       '3XL': 1024 * 256}
    default_file_name = {'L': 'cnnsum_L_16k.jsonl', 
                         'XL': 'cnnsum_XL_32k.jsonl', 
                         '2XL': 'cnnsum_2XL_64k.jsonl', 
                         '3XL': 'cnnsum_3XL_128k.jsonl', }

    context_len = max([default_max_len[k] for k in FLAGS.subsets])
    context_len = context_len if FLAGS.vllm_model_len is None else FLAGS.vllm_model_len
    data_path = './data/' if FLAGS.data_path is None else FLAGS.data_path

    model = LLM(FLAGS.base_model,
                rope_scaling=rope_scaling,
                cpu_offload_gb=FLAGS.cpu_offload_gb,
                skip_tokenizer_init=False,
                tensor_parallel_size=FLAGS.tensor_parallel_size,
                dtype='bfloat16',
                trust_remote_code=True, 
                gpu_memory_utilization=FLAGS.gpu_memory_utilization,
                max_model_len=context_len)
    
    with open('./prompts.json', 'r', encoding='utf-8') as tmp:
        prompts = json.load(tmp)

    model_outputs = {}
    all_scores = []
    for p in prompts:
        prompt = prompts[p]
        logger.info('Current Prompt: {}'.format(p))
        scores = []
        cur_prompt_outputs = {}
        for s in FLAGS.subsets:
            logger.info('Current CNNSum Subsets: {}'.format(s))
            sampling_params = SamplingParams(temperature=FLAGS.temperature, 
                                             top_k=FLAGS.top_k, top_p=FLAGS.top_p,
                                             max_tokens=default_infer_len[s],
                                             min_tokens=10,
                                             skip_special_tokens=True,
                                             use_beam_search=False)
            
            with jsonlines.open(data_path + default_file_name[s], 'r') as tmp:
                test_data = [{'context': d['context'], 'reference': d['summary']} for d in tmp]
            
            references = []
            predictions = []
            for data in tqdm(test_data):
                sources = prompt.format(data['context'])
                references.append(data['reference'])
                outputs = model.generate(prompts=sources, 
                                        sampling_params=sampling_params, 
                                        use_tqdm=False)[0]
                prompt_len = len(outputs.prompt_token_ids)
                output_len = len(outputs.outputs[0].token_ids)
                logger.info('input length: {} | output length: {}'.format(prompt_len, output_len))
                output = outputs.outputs[0].text
                predictions.append(output)

            score, _score = scorer(predictions, references)
            scores.append(score)
            logger.info('Subsets: {}, Prompt: {}, Score: {}'.format(s, p, score))
            
            cur_prompt_outputs[s] = predictions
        all_scores.append(scores)
    model_outputs[p] = cur_prompt_outputs
    all_scores = np.average(np.array(all_scores), axis=0)

    logger.info('final_scores:\n')
    for j in range(len(FLAGS.subsets)):
        logger.info('{}: {}'.format(FLAGS.subsets[j], all_scores[j]))

    with open(FLAGS.output_path, 'w', encoding='utf-8') as tmp:
        json.dump(model_outputs, tmp, ensure_ascii=False)


if __name__ == '__main__':
    app.run(main)