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

    infovqa_labels = np.load('answers_infovqa.npy', allow_pickle=True).item()
    okvqa_labels = np.load('answers_okvqa.npy', allow_pickle=True).item()

    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 
    models = [args.model]
    temp = args.variants    
    constant_T = 1

    for model_name in models:

        #if model_name + '_metrics_rephrase_output_'+str(temp)+'.npy' in os.listdir('.'):
        #    continue
        output_path = model_name + '_rephrase_output_'+str(temp)+'/'

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
            

            if file_name in infovqa_labels:
                gt = infovqa_labels[file_name]
            else:
                gt = okvqa_labels[file_name]

            for ggg in gt:
                ggg = ggg.lower()
                if ggg in results_0:
                    correct[count, 0] = 1
                if ggg in results_2:
                    correct[count, 1] = 1
                if ggg in results_2:
                    correct[count, 2] = 1

            ## Levenshtein Distance GT
            best_0 = 0
            best_1 = 0
            best_2 = 0
            for ggg in gt:
                ggg = ggg.lower()
                Levenshtein_distance_gt[count, 0] = max(fuzz.token_set_ratio(ggg, results_0), best_0)
                best_0 = max(fuzz.token_set_ratio(ggg, results_0), best_0)
                Levenshtein_distance_gt[count, 1] = max(fuzz.token_set_ratio(ggg, results_1), best_1)
                best_1 = max(fuzz.token_set_ratio(ggg, results_1), best_1)
                Levenshtein_distance_gt[count, 2] = max(fuzz.token_set_ratio(ggg, results_2), best_2)
                best_2 = max(fuzz.token_set_ratio(ggg, results_2), best_2)

            ## sentence similarity
            embedding_0 = sentence_model.encode(results_0, convert_to_tensor=True)
            embedding_1 = sentence_model.encode(results_1, convert_to_tensor=True)
            embedding_2 = sentence_model.encode(results_2, convert_to_tensor=True)
            best_0 = 0
            best_1 = 0
            best_2 = 0
            for ggg in gt:
                ggg = ggg.lower()
                embedding_gt = sentence_model.encode(ggg, convert_to_tensor=True)
                consistent_gt[count, 0] = max(float(util.pytorch_cos_sim(embedding_0, embedding_gt)), best_0)
                consistent_gt[count, 1] = max(float(util.pytorch_cos_sim(embedding_1, embedding_gt)), best_1)
                consistent_gt[count, 2] = max(float(util.pytorch_cos_sim(embedding_2, embedding_gt)), best_2)
                best_0 = max(float(util.pytorch_cos_sim(embedding_0, embedding_gt)), best_0)
                best_1 = max(float(util.pytorch_cos_sim(embedding_1, embedding_gt)), best_1)
                best_2 = max(float(util.pytorch_cos_sim(embedding_2, embedding_gt)), best_2)

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

    ## soft correctness 2
    #LLava_simgt = np.mean(LLava_sim_gt>=(gt_threshold/100))
    #print(model_name+'_soft_correct (SBERT similarity):', LLava_simgt*100)

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
