import numpy as np
import os
import argparse

from sentence_transformers import SentenceTransformer, util

from Levenshtein import distance
import statistics

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def find_best_answers(final_output_answers, pseudo_labels):
            
    def similarity(s1, s2):
        return fuzz.token_set_ratio(s1, s2) / 100.0
                
    best_avg_sim = -1
    best_set = None
    count = 0   
    for answers, pseudo_label in zip(final_output_answers, pseudo_labels):
        candidate = list(answers) + [pseudo_label]
        sims = []
        for i in range(len(candidate)):
            for j in range(i+1, len(candidate)):
                sims.append(similarity(candidate[i], candidate[j]))
        avg_sim = sum(sims) / len(sims)
        if avg_sim > best_avg_sim:
            best_avg_sim = avg_sim
            best_set = candidate
            best_step = count
        count += 1
    return best_set, best_step

def find_similar_strings(strings, threshold=0.85):
    """     
    Group similar strings together using Levenshtein distance.
    threshold: similarity threshold (0 to 1), higher means more similar
    """    
    groups = []
    used = set()

    for i, s1 in enumerate(strings):
        if i in used:
            continue

        current_group = [s1]
        used.add(i)

        # Compare with remaining strings
        for j, s2 in enumerate(strings[i+1:], start=i+1):
            if j in used:
                continue

            # Calculate similarity ratio
            max_len = max(len(s1), len(s2))
            if max_len == 0:
                similarity = 1.0 if s1 == s2 else 0.0
            else:
                similarity = 1 - (distance(s1, s2) / max_len)

            if similarity >= threshold:
                current_group.append(s2)
                used.add(j)

        groups.append(current_group)

    return groups

def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-m', '--model', type=str, help='model name (llava, llava_next, qwen-vl, InternVL3, Idefics2)')
    parser.add_argument('-v', '--variants', type=str, help='testing variants')
    args = parser.parse_args()

    indoor_labels = np.load('data/styling_task/indoor_labels.npy', allow_pickle=True).item()
    landmarks_labels = np.load('data/styling_task/landmarks_labels.npy', allow_pickle=True).item()

    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 

    models = [args.model]
    temp = args.variants

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

            output_text_0 = str(np.load(output_path+file_name+'_ori_0.npy', allow_pickle=True)).replace('</s>','').lower()
            output_text_1 = str(np.load(output_path+file_name+'_ori_1.npy', allow_pickle=True)).replace('</s>','').lower()
            output_text_2 = str(np.load(output_path+file_name+'_ori_2.npy', allow_pickle=True)).replace('</s>','').lower()
            output_text_3 = str(np.load(output_path+file_name+'_ori_3.npy', allow_pickle=True)).replace('</s>','').lower()

            similar_groups = find_similar_strings([output_text_0, output_text_1, output_text_2, output_text_3])
            largest_group = max(similar_groups, key=len)
            pseudo_label = largest_group[0]

            ad_step0_result_0 = str(np.load(output_path+file_name+'_step_0_0.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step0_result_1 = str(np.load(output_path+file_name+'_step_0_1.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step0_result_2 = str(np.load(output_path+file_name+'_step_0_2.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step0_result_3 = str(np.load(output_path+file_name+'_step_0_3.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step1_result_0 = str(np.load(output_path+file_name+'_step_1_0.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step1_result_1 = str(np.load(output_path+file_name+'_step_1_1.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step1_result_2 = str(np.load(output_path+file_name+'_step_1_2.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step1_result_3 = str(np.load(output_path+file_name+'_step_1_3.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step2_result_0 = str(np.load(output_path+file_name+'_step_2_0.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step2_result_1 = str(np.load(output_path+file_name+'_step_2_1.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step2_result_2 = str(np.load(output_path+file_name+'_step_2_2.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step2_result_3 = str(np.load(output_path+file_name+'_step_2_3.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step3_result_0 = str(np.load(output_path+file_name+'_step_3_0.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step3_result_1 = str(np.load(output_path+file_name+'_step_3_1.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step3_result_2 = str(np.load(output_path+file_name+'_step_3_2.npy', allow_pickle=True)).replace('</s>','').lower()
            ad_step3_result_3 = str(np.load(output_path+file_name+'_step_3_3.npy', allow_pickle=True)).replace('</s>','').lower()

            final_output_answers = []
            final_output_answers.append([ad_step0_result_0, ad_step0_result_1, ad_step0_result_2, ad_step0_result_3])
            final_output_answers.append([ad_step1_result_0, ad_step1_result_1, ad_step1_result_2, ad_step1_result_3])
            final_output_answers.append([ad_step2_result_0, ad_step2_result_1, ad_step2_result_2, ad_step2_result_3])
            final_output_answers.append([ad_step3_result_0, ad_step3_result_1, ad_step3_result_2, ad_step3_result_3])
            _, best_step = find_best_answers(final_output_answers, [pseudo_label] * 4)

            results_0 = str(np.load(output_path+file_name+'_step_'+str(best_step)+'_eval_0.npy', allow_pickle=True)).replace('</s>','').lower()
            results_1 = str(np.load(output_path+file_name+'_step_'+str(best_step)+'_eval_1.npy', allow_pickle=True)).replace('</s>','').lower()
            results_2 = str(np.load(output_path+file_name+'_step_'+str(best_step)+'_eval_2.npy', allow_pickle=True)).replace('</s>','').lower()
            results_3 = str(np.load(output_path+file_name+'_step_'+str(best_step)+'_eval_3.npy', allow_pickle=True)).replace('</s>','').lower()

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
    # soft correctness
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
