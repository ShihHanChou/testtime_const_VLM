import numpy as np
import os
import argparse

from sentence_transformers import SentenceTransformer, util

import statistics

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def main():

    indoor_labels = np.load('indoor_labels.npy', allow_pickle=True).item()
    landmarks_labels = np.load('landmarks_labels.npy', allow_pickle=True).item()

    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-m', '--model', type=str, help='model name (llava, llava_next, qwen-vl, InternVL3, Idefics2)')
    parser.add_argument('-v', '--variants', type=str, help='testing variants')
    args = parser.parse_args()

    models = [args.model]
    temp = args.variants
    constant_T = 1

    for model_name in models:

        output_path = model_name + '_restyle_output_'+str(temp)+'/'

        eval_list = []
        for ooo in os.listdir(output_path):
            if '_ori_0.npy' not in ooo:
                continue
            eval_name = ooo.split('_ori_0.npy')[0]
            if eval_name not in eval_list:
                eval_list.append(eval_name)

        data_number = len(eval_list)

        correct = np.zeros((data_number, 4))
        Levenshtein_distance_gt = np.zeros((data_number, 4))
        consistent = np.zeros((data_number, 6))
        consistent_gt = np.zeros((data_number, 4))
        type_style = []
        count=0
        for iii, ele in enumerate(eval_list):

            print(count, data_number)

            file_name = ele

            results_0 = str(np.load(output_path+file_name+'_step_'+str(constant_T)+'_eval_0.npy', allow_pickle=True)).replace('</s>','').lower()
            results_1 = str(np.load(output_path+file_name+'_step_'+str(constant_T)+'_eval_1.npy', allow_pickle=True)).replace('</s>','').lower()
            results_2 = str(np.load(output_path+file_name+'_step_'+str(constant_T)+'_eval_2.npy', allow_pickle=True)).replace('</s>','').lower()
            results_3 = str(np.load(output_path+file_name+'_step_'+str(constant_T)+'_eval_3.npy', allow_pickle=True)).replace('</s>','').lower()

            if ele in indoor_labels:
                gt = indoor_labels[ele]
            else:
                gt = landmarks_labels[ele]

            if gt in results_0:
                correct[count, 0] = 1
            if gt in results_1:
                correct[count, 1] = 1
            if gt in results_2:
                correct[count, 2] = 1
            if gt in results_3:
                correct[count, 3] = 1

            ## Levenshtein Distance GT
            Levenshtein_distance_gt[count, 0] = fuzz.token_set_ratio(gt.lower(), results_0)
            Levenshtein_distance_gt[count, 1] = fuzz.token_set_ratio(gt.lower(), results_1)
            Levenshtein_distance_gt[count, 2] = fuzz.token_set_ratio(gt.lower(), results_2)
            Levenshtein_distance_gt[count, 3] = fuzz.token_set_ratio(gt.lower(), results_3)

            ## sentence similarity
            embedding_0 = sentence_model.encode(results_0, convert_to_tensor=True)
            embedding_1 = sentence_model.encode(results_1, convert_to_tensor=True)
            embedding_2 = sentence_model.encode(results_2, convert_to_tensor=True)
            embedding_3 = sentence_model.encode(results_3, convert_to_tensor=True)

            embedding_gt = sentence_model.encode(gt, convert_to_tensor=True)
            consistent_gt[count, 0] = float(util.pytorch_cos_sim(embedding_0, embedding_gt))
            consistent_gt[count, 1] = float(util.pytorch_cos_sim(embedding_1, embedding_gt))
            consistent_gt[count, 2] = float(util.pytorch_cos_sim(embedding_2, embedding_gt))
            consistent_gt[count, 3] = float(util.pytorch_cos_sim(embedding_3, embedding_gt))

            ## sentence similarity
            consistent[count, 0] = float(util.pytorch_cos_sim(embedding_0, embedding_1))
            consistent[count, 1] = float(util.pytorch_cos_sim(embedding_0, embedding_2))
            consistent[count, 2] = float(util.pytorch_cos_sim(embedding_0, embedding_3))
            consistent[count, 3] = float(util.pytorch_cos_sim(embedding_1, embedding_2))
            consistent[count, 4] = float(util.pytorch_cos_sim(embedding_1, embedding_3))
            consistent[count, 5] = float(util.pytorch_cos_sim(embedding_2, embedding_3))

            count += 1

        output = {} 
        output['correct'] = correct[:count]
        output['Levenshtein_distance_gt'] = Levenshtein_distance_gt[:count]
        output['consistent_gt'] = consistent_gt[:count]
        output['consistent'] = consistent[:count]

        np.save(model_name + '_metrics_style_output_'+str(temp)+'.npy', output)

    print(model_name + '_metrics_style_output_'+str(temp))
    LLava_corr = output['correct']
    LLava_Levenshtein_distance_gt = output['Levenshtein_distance_gt']
    LLava_sim_gt = output['consistent_gt']
    LLava_sim = output['consistent']

    gt_threshold = 85
    # soft correctness 1
    LLava_ldgt = np.mean(LLava_Levenshtein_distance_gt>=gt_threshold)
    print(model_name+'_soft_correct (fuzzywuzzy):', LLava_ldgt*100)

    # sentence similarity gt
    LLava_ssgt = np.mean(LLava_sim_gt)
    print(model_name+'_ss_gt:', LLava_ssgt*100)

    # consistent
    threshold = 0.7
    LLava_con = np.mean(LLava_sim>threshold)
    print(model_name+'_consistency:', LLava_con*100)

    # sentence similarity
    LLava_ss = np.mean(LLava_sim)
    print(model_name+'_ss:', LLava_ss*100)

    H_mean = statistics.harmonic_mean([(LLava_ldgt*100+LLava_ssgt*100)/2, (LLava_con*100+LLava_ss*100)/2])
    print(model_name+'_Hmean:', H_mean)

if __name__ == '__main__':
    main()
