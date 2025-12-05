import numpy as np
import os
import argparse

from sentence_transformers import SentenceTransformer, util

from Levenshtein import distance
import statistics

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def main():

    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-m', '--model', type=str, help='model name (llava, llava_next, qwen-vl, InternVL3, Idefics2)')
    parser.add_argument('-v', '--variants', type=str, help='testing variants')
    args = parser.parse_args()

    categories = np.load('data/masking_task/coco_category_dic.npy', allow_pickle=True).item()
    types = np.load('data/masking_task/types.npy', allow_pickle=True).item()
    labels = np.load('data/masking_task/labels.npy', allow_pickle=True).item()

    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


    models = [args.model]
    temp = args.variants
    constant_T = 1

    for model_name in models:

        output_path = model_name + '_masking_output_'+str(temp)+'/'

        eval_list = []
        for ooo in os.listdir(output_path):
            eval_name = ooo.split('_')[0] + '_' + ooo.split('_')[1]
            if eval_name not in eval_list:
                eval_list.append(eval_name)

        data_number = len(eval_list)

        correct = np.zeros((data_number, 3))
        Levenshtein_distance_gt = np.zeros((data_number, 3))
        consistent = np.zeros((data_number, 3))
        consistent_gt = np.zeros((data_number, 3))
        pairwise_stats_all = []

        count=0
        for iii, ele in enumerate(eval_list):
            print(count, data_number)

            file_name = ele

            results_0 = str(np.load(output_path+file_name+'_step_'+str(constant_T)+'_eval_0.npy', allow_pickle=True)).replace('</s>','').lower()
            results_1 = str(np.load(output_path+file_name+'_step_'+str(constant_T)+'_eval_1.npy', allow_pickle=True)).replace('</s>','').lower()
            results_2 = str(np.load(output_path+file_name+'_step_'+str(constant_T)+'_eval_2.npy', allow_pickle=True)).replace('</s>','').lower()

            results_0 = results_0.replace(',', '')
            results_0 = results_0.replace('.', '')
            results_0 = results_0.replace('in the masked region', '')
            results_0 = results_0.replace('the masked region', '')
            results_0 = results_0.replace('there is ', '')
            results_0 = results_0.replace('is ', '')
            results_0 = results_0.replace('contains ', '')
            results_0 = results_0.replace('</s>', '')
            results_0 = results_0.replace('the ', '')
            results_0 = results_0.replace('in ', '')
            results_0 = results_0.replace('of ', '')
            results_0 = results_0.replace('image', '')
            results_0 = results_0.replace(' a ', ' ')
            results_0 = results_0.replace(' an ', ' ')

            results_1 = results_1.replace(',', '')
            results_1 = results_1.replace('.', '')
            results_1 = results_1.replace('in the masked region', '')
            results_1 = results_1.replace('the masked region', '')
            results_1 = results_1.replace('there is ', '')
            results_1 = results_1.replace('is ', '')
            results_1 = results_1.replace('contains ', '')
            results_1 = results_1.replace('</s>', '')
            results_1 = results_1.replace('the ', '')
            results_1 = results_1.replace('in ', '')
            results_1 = results_1.replace('of ', '')
            results_1 = results_1.replace('image', '')
            results_1 = results_1.replace(' a ', ' ')
            results_1 = results_1.replace(' an ', ' ')
            
            results_2 = results_2.replace(',', '')
            results_2 = results_2.replace('.', '')
            results_2 = results_2.replace('in the masked region', '')
            results_2 = results_2.replace('the masked region', '')
            results_2 = results_2.replace('there is ', '')
            results_2 = results_2.replace('is ', '')
            results_2 = results_2.replace('contains ', '')
            results_2 = results_2.replace('</s>', '')
            results_2 = results_2.replace('the ', '')
            results_2 = results_2.replace('in ', '')
            results_2 = results_2.replace('of ', '')
            results_2 = results_2.replace('image', '')
            results_2 = results_2.replace(' a ', ' ')
            results_2 = results_2.replace(' an ', ' ')

            if len(results_0) > 0:
                if results_0[0] == ' ':
                    results_0 = results_0[1:]
            if len(results_1) > 0:
                if results_1[0] == ' ':
                    results_1 = results_1[1:]
            if len(results_2) > 0:
                if results_2[0] == ' ':
                    results_2 = results_2[1:]

            gt = categories[labels[file_name+'_0']]

            if gt in results_0:
                correct[count, 0] = 1
            if gt in results_2:
                correct[count, 1] = 1
            if gt in results_2:
                correct[count, 2] = 1

            ## Levenshtein Distance GT
            Levenshtein_distance_gt[count, 0] = max(fuzz.token_set_ratio(gt, results_0), best_0)
            Levenshtein_distance_gt[count, 1] = max(fuzz.token_set_ratio(gt, results_1), best_1)
            Levenshtein_distance_gt[count, 2] = max(fuzz.token_set_ratio(gt, results_2), best_2)

            ## sentence similarity
            embedding_0 = sentence_model.encode(results_0, convert_to_tensor=True)
            embedding_1 = sentence_model.encode(results_1, convert_to_tensor=True)
            embedding_2 = sentence_model.encode(results_2, convert_to_tensor=True)
            embedding_gt = sentence_model.encode(gt, convert_to_tensor=True)
            consistent_gt[count, 0] = max(float(util.pytorch_cos_sim(embedding_0, embedding_gt)), best_0)
            consistent_gt[count, 1] = max(float(util.pytorch_cos_sim(embedding_1, embedding_gt)), best_1)
            consistent_gt[count, 2] = max(float(util.pytorch_cos_sim(embedding_2, embedding_gt)), best_2)

            ## sentence similarity
            consistent[count, 0] = float(util.pytorch_cos_sim(embedding_0, embedding_1))
            consistent[count, 1] = float(util.pytorch_cos_sim(embedding_0, embedding_2))
            consistent[count, 2] = float(util.pytorch_cos_sim(embedding_1, embedding_2))

            count += 1

        output = {} 
        output['correct'] = correct[:count]
        output['Levenshtein_distance_gt'] = Levenshtein_distance_gt[:count]
        output['consistent_gt'] = consistent_gt[:count]
        output['consistent'] = consistent[:count]

        np.save(model_name + '_metrics_rephrase_output_'+str(temp)+'.npy', output)

    output = np.load(model_name + '_metrics_rephrase_output_'+str(temp)+'.npy', allow_pickle=True).item()

    print(model_name + '_metrics_rephrase_output_'+str(temp))
    LLava_corr = output['correct']
    LLava_Levenshtein_distance_gt = output['Levenshtein_distance_gt']
    LLava_sim_gt = output['consistent_gt']
    LLava_sim = output['consistent']

    # correctness
    LLava_c = np.mean(LLava_corr)
    print(model_name+'_correct:', LLava_c*100)

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

    H_mean = statistics.harmonic_mean([(LLava_c*100+LLava_ssgt*100)/2, (LLava_con*100+LLava_ss*100)/2])
    print(model_name+'_Hmean:', H_mean)

if __name__ == '__main__':
    main()
