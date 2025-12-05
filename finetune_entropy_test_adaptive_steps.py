import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json
from PIL import Image
from Levenshtein import distance

from finetune_data_entropy_test_hf import RephraseDataset, StyleDataset, MaskingDataset

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x1, x2):
        # Apply softmax to convert logits to probability distributions
        p1 = F.softmax(x1, dim=-1)
        p2 = F.softmax(x2, dim=-1)
        
        # Compute cross entropy in both directions
        # We use p1 as target for x2 and p2 as target for x1
        ce_1_2 = F.cross_entropy(x1, p2)
        ce_2_1 = F.cross_entropy(x2, p1)
        
        # Average the cross entropy losses
        return (ce_1_2 + ce_2_1) / 2.0

class PseudoLabelLoss(nn.Module):
    def __init__(self):
        super(PseudoLabelLoss, self).__init__()
    
    def forward(self, logits, target_ids):
        # Shift the logits and targets for language modeling
        # We predict the next token given the previous tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the logits to (batch_size * sequence_length, vocab_size)
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        # Flatten the labels to (batch_size * sequence_length,)
        shift_labels = shift_labels.view(-1)
        
        return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

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

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                        help='evaluate model on test set')
    parser.add_argument('-m', '--model', type=str, help='model name (llava, llava_next, qwen-vl)')
    parser.add_argument('-t', '--task', type=str, help='task (rephrase, restyle, masking)')
    parser.add_argument('-n', '--number', type=int, help='An integer number')
    parser.add_argument('-a', '--alpha', type=float, help='A float number')
    parser.add_argument('-b', '--beta', type=float, help='A float number')
    parser.add_argument('-u', '--update_steps', type=int, help='Update steps')
    args = parser.parse_args()

    evaluate = args.evaluate
    arg_index = args.number
    arg_alpha = args.alpha
    arg_beta = args.beta
    model_name = args.model
    task = args.task
    update_steps = args.update_steps

    if model_name == "llava":
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        model_id = "llava-hf/llava-1.5-7b-hf"
        llava_model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,)
        processor = AutoProcessor.from_pretrained(model_id)
    elif model_name == "llava_next":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    elif model_name == "qwen-vl":
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        min_pixels = 256*28*28
        max_pixels = 1024*28*28 
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
        llava_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct",device_map="auto")
    elif model_name == "InternVL3":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        torch_device = "cuda"
        model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
        processor = AutoProcessor.from_pretrained(model_checkpoint)
        llava_model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)
    elif model_name == "Idefics2":
        from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
        torch_device = "cuda"
        model_checkpoint = "HuggingFaceM4/idefics2-8b"
        processor = Idefics2Processor.from_pretrained(model_checkpoint)
        llava_model = Idefics2ForConditionalGeneration.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)
    llava_model.to("cuda:0")
    llava_model = llava_model.to(torch.float32)

    batchsize = 1
    if task == "rephrase":
        folder_root = 'data/rephrase_task/'
        img_root = folder_root + 'rephrase_task_images/'
        text_root_train = folder_root + 'rephrase_questions_eval/'
        train_set = RephraseDataset(folder_root, img_root, text_root_train, llava_model, processor, evaluate, arg_index, model_name)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=False)
    elif task == "restyle":
        folder_root = 'data/styling_task/'
        img_root = folder_root + 'generated_style/'
        text_root_train = ''
        train_set = StyleDataset(folder_root, img_root, text_root_train, llava_model, processor, evaluate, arg_index, model_name)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=False)
    elif task == "masking":
        folder_root = 'data/masking_task/'
        img_root = folder_root + 'generated/'
        text_root_train = ''
        train_set = MaskingDataset(folder_root, img_root, text_root_train, llava_model, processor, evaluate, arg_index, model_name)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=False)

    output_folder_path = folder_root + model_name+'_'+task+'_output_entropy_test_5e4_adaptivesteps_fuzzywuzzy_en'+str(arg_alpha)+'_plabel'+str(arg_beta)+'/'
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    lr = 5e-4
    decay_rate = 0.98

    # Only get parameters from projector and decoder
    parameters = []
    for name, param in llava_model.named_parameters():
        # Tune the projector parameters
        if 'lm_head' in name:
            param.requires_grad = True
            parameters.append(param)
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW([p for p in parameters if p.requires_grad], lr=lr, weight_decay=0.01)
    print('Number of trainable parameters:', sum(p.numel() for p in parameters if p.requires_grad))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    entropy_loss = HLoss()
    entropy_update(train_loader, llava_model, optimizer, parameters, entropy_loss, processor, output_folder_path, arg_alpha, arg_beta, model_name, task, update_steps, args)

def get_text_outputs(llava_model, processor, inputs, args):
    temperature = 0.0
    inputs = inputs.to(llava_model.device)
    with torch.inference_mode():
        if args.model == "qwen-vl":
            output = llava_model.generate(**inputs, max_new_tokens=100, do_sample=False)
        else:
            output = llava_model.generate(**inputs, max_new_tokens=100, temperature = temperature)
        generated_answers = processor.decode(output[0], skip_special_tokens=True)
    return generated_answers

def find_best_answers(final_output_answers, pseudo_labels):
    from fuzzywuzzy import fuzz

    def similarity(s1, s2):
        return fuzz.token_set_ratio(s1, s2) / 100.0

    best_avg_sim = -1
    best_set = None
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
    return best_set

def entropy_update(train_loader, llava_model, optimizer, train_param, entropy_loss, processor, output_folder_path, arg_alpha, arg_beta, model_name, task, update_steps,args):

    # switch to train mode
    loss_epoch = []

    llava_model.train()
    #update_steps = 10
    alpha = arg_alpha
    beta = arg_beta
    pseudo_label_loss = PseudoLabelLoss()

    for i, (image_sizes, input_ids, targets, file_name, o_input_ids) in enumerate(train_loader):

        if i > 0:
            break
        print(i)

        ''' model & full_model'''
        input_ids = input_ids.squeeze(0).to(llava_model.device)
        targets = targets.squeeze(0).to(llava_model.device)
        image_sizes = [(int(image_sizes[0][0]), int(image_sizes[0][1]))]

        ## generate original answers
        targets_tmp = targets[targets>0]
        o_answer = processor.tokenizer.batch_decode(targets_tmp, skip_special_tokens=True)
        print('o_answer: ', o_answer)

        if task == 'restyle':
            image_sizes = [image_sizes[0], image_sizes[0], image_sizes[0], image_sizes[0]]
            for kkk in o_input_ids[0].keys():
                o_input_ids[0][kkk] = o_input_ids[0][kkk].squeeze(0)
            for kkk in o_input_ids[1].keys():
                o_input_ids[1][kkk] = o_input_ids[1][kkk].squeeze(0)
            for kkk in o_input_ids[2].keys():
                o_input_ids[2][kkk] = o_input_ids[2][kkk].squeeze(0)
            for kkk in o_input_ids[3].keys():
                o_input_ids[3][kkk] = o_input_ids[3][kkk].squeeze(0)
            output_text_0 = get_text_outputs(llava_model, processor, o_input_ids[0], args)
            output_text_1 = get_text_outputs(llava_model, processor, o_input_ids[1], args)
            output_text_2 = get_text_outputs(llava_model, processor, o_input_ids[2], args)
            output_text_4 = get_text_outputs(llava_model, processor, o_input_ids[3], args)
            if model_name == "llava":
                output_text_0 = output_text_0.split('ASSISTANT: ')[1]
                output_text_1 = output_text_1.split('ASSISTANT: ')[1]
                output_text_2 = output_text_2.split('ASSISTANT: ')[1]
                output_text_4 = output_text_4.split('ASSISTANT: ')[1]
            elif model_name == "llava_next":
                output_text_0 = output_text_0.split('[/INST]')[1][1:-1]
                output_text_1 = output_text_1.split('[/INST]')[1][1:-1]
                output_text_2 = output_text_2.split('[/INST]')[1][1:-1]
                output_text_4 = output_text_4.split('[/INST]')[1][1:-1]
            elif model_name == "qwen-vl":
                output_text_0 = output_text_0.split('assistant\n')[-1]
                output_text_1 = output_text_1.split('assistant\n')[-1]
                output_text_2 = output_text_2.split('assistant\n')[-1]
                output_text_4 = output_text_4.split('assistant\n')[-1]
            elif model_name == "InternVL3":
                output_text_0 = output_text_0.split('assistant\n')[-1]
                output_text_1 = output_text_1.split('assistant\n')[-1]
                output_text_2 = output_text_2.split('assistant\n')[-1]
                output_text_4 = output_text_4.split('assistant\n')[-1]
            elif model_name == "Idefics2":
                output_text_0 = output_text_0.split('Assistant: ')[-1]
                output_text_1 = output_text_1.split('Assistant: ')[-1]
                output_text_2 = output_text_2.split('Assistant: ')[-1]
                output_text_4 = output_text_4.split('Assistant: ')[-1]
            np.save(output_folder_path + file_name[0] + '_ori_0.npy', output_text_0)
            np.save(output_folder_path + file_name[0] + '_ori_1.npy', output_text_1)
            np.save(output_folder_path + file_name[0] + '_ori_2.npy', output_text_2)
            np.save(output_folder_path + file_name[0] + '_ori_3.npy', output_text_4) 
            print('answer_1: ', output_text_0, 'answer_2: ', output_text_1, 'answer_3: ', output_text_2, 'answer_4: ', output_text_4)
            similar_groups = find_similar_strings([output_text_0, output_text_1, output_text_2, output_text_4])
            largest_group = max(similar_groups, key=len)
            pseudo_label = largest_group[0]
            print('pseudo_label: ', pseudo_label)

            pseudo_label_tokens = processor.tokenizer(pseudo_label, return_tensors="pt").input_ids.to(llava_model.device)
            # Encode pseudo-label for loss computation
            targets = torch.full_like(input_ids, -100)
            pad_len = 10
            if pseudo_label_tokens.shape[1] >= pad_len:
                targets[:,-pad_len:] = pseudo_label_tokens[:,:pad_len]
            else:
                targets[:,-pad_len:-pad_len+pseudo_label_tokens.shape[1]] = pseudo_label_tokens

            print('total update steps: ', update_steps)
            final_output_answers = []
            for step in range(update_steps):
                print(step)
                print('updating model: ', step)
                ## entropy updates
                if model_name == "qwen-vl":
                    outputs_0 = llava_model(input_ids=input_ids[0].unsqueeze(0), labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_1 = llava_model(input_ids=input_ids[1].unsqueeze(0), labels=targets[1].unsqueeze(0), return_dict=True)
                    outputs_2 = llava_model(input_ids=input_ids[2].unsqueeze(0), labels=targets[2].unsqueeze(0), return_dict=True)
                    outputs_3 = llava_model(input_ids=input_ids[3].unsqueeze(0), labels=targets[3].unsqueeze(0), return_dict=True)
                elif model_name == "InternVL3":
                    outputs_0 = llava_model(input_ids=input_ids[0].unsqueeze(0), labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_1 = llava_model(input_ids=input_ids[1].unsqueeze(0), labels=targets[1].unsqueeze(0), return_dict=True)
                    outputs_2 = llava_model(input_ids=input_ids[2].unsqueeze(0), labels=targets[2].unsqueeze(0), return_dict=True)
                    outputs_3 = llava_model(input_ids=input_ids[3].unsqueeze(0), labels=targets[3].unsqueeze(0), return_dict=True)
                elif model_name == "Idefics2":
                    outputs_0 = llava_model(input_ids=input_ids[0].unsqueeze(0), labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_1 = llava_model(input_ids=input_ids[1].unsqueeze(0), labels=targets[1].unsqueeze(0), return_dict=True)
                    outputs_2 = llava_model(input_ids=input_ids[2].unsqueeze(0), labels=targets[2].unsqueeze(0), return_dict=True)
                    outputs_3 = llava_model(input_ids=input_ids[3].unsqueeze(0), labels=targets[3].unsqueeze(0), return_dict=True)
                elif model_name == "llava_next":
                    outputs_0 = llava_model(input_ids=input_ids[0].unsqueeze(0), images=o_input_ids[0]['pixel_values'].to(torch.float32), image_sizes=image_sizes[0], labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_1 = llava_model(input_ids=input_ids[1].unsqueeze(0), images=o_input_ids[1]['pixel_values'].to(torch.float32), image_sizes=image_sizes[1], labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_2 = llava_model(input_ids=input_ids[2].unsqueeze(0), images=o_input_ids[2]['pixel_values'].to(torch.float32), image_sizes=image_sizes[2], labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_3 = llava_model(input_ids=input_ids[3].unsqueeze(0), images=o_input_ids[3]['pixel_values'].to(torch.float32), image_sizes=image_sizes[3], labels=targets[0].unsqueeze(0), return_dict=True)
                elif model_name == "llava":
                    outputs_0 = llava_model(input_ids=input_ids[0].unsqueeze(0), images=o_input_ids[0]['pixel_values'].to(torch.float32), image_sizes=image_sizes[0], labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_1 = llava_model(input_ids=input_ids[1].unsqueeze(0), images=o_input_ids[1]['pixel_values'].to(torch.float32), image_sizes=image_sizes[1], labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_2 = llava_model(input_ids=input_ids[2].unsqueeze(0), images=o_input_ids[2]['pixel_values'].to(torch.float32), image_sizes=image_sizes[2], labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_3 = llava_model(input_ids=input_ids[3].unsqueeze(0), images=o_input_ids[3]['pixel_values'].to(torch.float32), image_sizes=image_sizes[3], labels=targets[0].unsqueeze(0), return_dict=True)

                # Clip extreme values in logits
                out_labels=outputs_0['labels']
                if model_name == "llava":
                    logits_1 = outputs_0['logits'][0][out_labels[0]!=-100][1:]
                    logits_2 = outputs_1['logits'][0][out_labels[0]!=-100][1:]
                    logits_3 = outputs_2['logits'][0][out_labels[0]!=-100][1:]
                    logits_4 = outputs_3['logits'][0][out_labels[0]!=-100][1:]
                elif model_name == "llava_next":
                    logits_1 = outputs_0['logits'][0][out_labels[0]!=-100][1:]
                    logits_2 = outputs_1['logits'][0][out_labels[0]!=-100][1:]
                    logits_3 = outputs_2['logits'][0][out_labels[0]!=-100][1:]
                    logits_4 = outputs_3['logits'][0][out_labels[0]!=-100][1:]
                elif model_name == "qwen-vl":
                    logits_1 = outputs_0['logits'][0][out_labels[0]!=-100]
                    logits_2 = outputs_1['logits'][0][out_labels[0]!=-100]
                    logits_3 = outputs_2['logits'][0][out_labels[0]!=-100]
                    logits_4 = outputs_3['logits'][0][out_labels[0]!=-100]
                elif model_name == "InternVL3":
                    logits_1 = outputs_0['logits'][0][out_labels[0]!=-100]
                    logits_2 = outputs_1['logits'][0][out_labels[0]!=-100]
                    logits_3 = outputs_2['logits'][0][out_labels[0]!=-100]
                    logits_4 = outputs_3['logits'][0][out_labels[0]!=-100]
                elif model_name == "Idefics2":
                    logits_1 = outputs_0['logits'][0][out_labels[0]!=-100][1:]
                    logits_2 = outputs_1['logits'][0][out_labels[0]!=-100][1:]
                    logits_3 = outputs_2['logits'][0][out_labels[0]!=-100][1:]
                    logits_4 = outputs_3['logits'][0][out_labels[0]!=-100][1:]
                logits_1 = torch.mean(logits_1, 0).unsqueeze(0)
                logits_2 = torch.mean(logits_2, 0).unsqueeze(0)
                logits_3 = torch.mean(logits_3, 0).unsqueeze(0)
                logits_4 = torch.mean(logits_4, 0).unsqueeze(0)
                loss_entorpy = (entropy_loss(logits_1, logits_2) + entropy_loss(logits_1, logits_3) + entropy_loss(logits_1, logits_4) + \
                entropy_loss(logits_2, logits_3) + entropy_loss(logits_2, logits_4) + entropy_loss(logits_3, logits_4))/6.0
                
                # Scale and stabilize logits
                loss_0 = outputs_0.loss
                loss_1 = outputs_1.loss
                loss_2 = outputs_2.loss
                loss_3 = outputs_3.loss
                loss_pseudo_label = (loss_0 + loss_1 + loss_2 + loss_3) / 4.0
                
                # Print statistics before loss computation
                loss = alpha * loss_entorpy + beta * loss_pseudo_label
                print(f"Pseudo Label Loss value: {loss_pseudo_label.item():.4f}")
                print(f"Entropy Loss value: {loss_entorpy.item():.4f}")
                print(f"Loss value: {loss.item():.4f}")
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Warning: Loss is NaN or Inf, skipping batch")
                    continue
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch.append(loss.item())
                    
                output_text_0 = get_text_outputs(llava_model, processor, o_input_ids[0], args)
                output_text_1 = get_text_outputs(llava_model, processor, o_input_ids[1], args)
                output_text_2 = get_text_outputs(llava_model, processor, o_input_ids[2], args)
                output_text_4 = get_text_outputs(llava_model, processor, o_input_ids[3], args)
                if model_name == "qwen-vl":
                    output_text_0 = output_text_0.split('assistant\n')[-1]
                    output_text_1 = output_text_1.split('assistant\n')[-1]
                    output_text_2 = output_text_2.split('assistant\n')[-1]
                    output_text_4 = output_text_4.split('assistant\n')[-1]
                elif model_name == "InternVL3":
                    output_text_0 = output_text_0.split('assistant\n')[-1]
                    output_text_1 = output_text_1.split('assistant\n')[-1]
                    output_text_2 = output_text_2.split('assistant\n')[-1]
                    output_text_4 = output_text_4.split('assistant\n')[-1]
                elif model_name == "Idefics2":
                    output_text_0 = output_text_0.split('Assistant: ')[-1]
                    output_text_1 = output_text_1.split('Assistant: ')[-1]
                    output_text_2 = output_text_2.split('Assistant: ')[-1]
                    output_text_4 = output_text_4.split('Assistant: ')[-1]
                elif model_name == "llava":
                    output_text_0 = output_text_0.split('ASSISTANT: ')[1]
                    output_text_1 = output_text_1.split('ASSISTANT: ')[1]
                    output_text_2 = output_text_2.split('ASSISTANT: ')[1]
                    output_text_4 = output_text_4.split('ASSISTANT: ')[1]
                elif model_name == "llava_next":
                    output_text_0 = output_text_0.split('[/INST]')[1][1:-1]
                    output_text_1 = output_text_1.split('[/INST]')[1][1:-1]
                    output_text_2 = output_text_2.split('[/INST]')[1][1:-1]
                    output_text_4 = output_text_4.split('[/INST]')[1][1:-1]
                print('answer_1: ', output_text_0, 'answer_2: ', output_text_1, 'answer_3: ', output_text_2, 'answer_4: ', output_text_4)
                final_output_answers.append([output_text_0, output_text_1, output_text_2, output_text_4])
                np.save(output_folder_path + file_name[0][:-2] + '_step_'+str(step)+'_0.npy', output_text_0)
                np.save(output_folder_path + file_name[0][:-2] + '_step_'+str(step)+'_1.npy', output_text_1)
                np.save(output_folder_path + file_name[0][:-2] + '_step_'+str(step)+'_2.npy', output_text_2)
                np.save(output_folder_path + file_name[0][:-2] + '_step_'+str(step)+'_3.npy', output_text_4)

        else:
            image_sizes = [image_sizes[0], image_sizes[0], image_sizes[0]]
            # Generate initial outputs and create pseudo-label
            for kkk in o_input_ids[0].keys():
                o_input_ids[0][kkk] = o_input_ids[0][kkk].squeeze(0)
            for kkk in o_input_ids[1].keys():
                o_input_ids[1][kkk] = o_input_ids[1][kkk].squeeze(0)
            for kkk in o_input_ids[2].keys():
                o_input_ids[2][kkk] = o_input_ids[2][kkk].squeeze(0)
            output_text_0 = get_text_outputs(llava_model, processor, o_input_ids[0], args)
            output_text_1 = get_text_outputs(llava_model, processor, o_input_ids[1], args)
            output_text_2 = get_text_outputs(llava_model, processor, o_input_ids[2], args)

            if model_name == "qwen-vl":
                output_text_0 = output_text_0.split('assistant\n')[-1]
                output_text_1 = output_text_1.split('assistant\n')[-1]
                output_text_2 = output_text_2.split('assistant\n')[-1]
            if model_name == "InternVL3":
                output_text_0 = output_text_0.split('assistant\n')[-1]
                output_text_1 = output_text_1.split('assistant\n')[-1]
                output_text_2 = output_text_2.split('assistant\n')[-1]
            if model_name == "Idefics2":
                output_text_0 = output_text_0.split('Assistant: ')[-1]
                output_text_1 = output_text_1.split('Assistant: ')[-1]
                output_text_2 = output_text_2.split('Assistant: ')[-1]
            elif model_name == "llava":
                output_text_0 = output_text_0.split('ASSISTANT: ')[1]
                output_text_1 = output_text_1.split('ASSISTANT: ')[1]
                output_text_2 = output_text_2.split('ASSISTANT: ')[1]
            elif model_name == "llava_next":
                output_text_0 = output_text_0.split('[/INST]')[1][1:-1]
                output_text_1 = output_text_1.split('[/INST]')[1][1:-1]
                output_text_2 = output_text_2.split('[/INST]')[1][1:-1]
            print('answer_1: ', output_text_0, 'answer_2: ', output_text_1, 'answer_3: ', output_text_2)
            similar_groups = find_similar_strings([output_text_0, output_text_1, output_text_2])
            largest_group = max(similar_groups, key=len)
            pseudo_label = largest_group[0]
            print('pseudo_label: ', pseudo_label)
            pseudo_label_tokens = processor.tokenizer(pseudo_label, return_tensors="pt").input_ids.to(llava_model.device)

            # Encode pseudo-label for loss computation
            targets = torch.full_like(input_ids, -100)
            pad_len = 10
            if pseudo_label_tokens.shape[1] >= pad_len:
                targets[:,-pad_len:] = pseudo_label_tokens[:,:pad_len]
            else:
                targets[:,-pad_len:-pad_len+pseudo_label_tokens.shape[1]] = pseudo_label_tokens
            print('total update steps: ', update_steps)
            final_output_answers = []

            for step in range(update_steps):
                print(step)
                print('updating model: ', step)
                ## entropy updates
                if model_name == "qwen-vl":
                    outputs_0 = llava_model(input_ids=input_ids[0].unsqueeze(0), labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_1 = llava_model(input_ids=input_ids[1].unsqueeze(0), labels=targets[1].unsqueeze(0), return_dict=True)
                    outputs_2 = llava_model(input_ids=input_ids[2].unsqueeze(0), labels=targets[2].unsqueeze(0), return_dict=True)
                elif model_name == "InternVL3":
                    outputs_0 = llava_model(input_ids=input_ids[0].unsqueeze(0), labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_1 = llava_model(input_ids=input_ids[1].unsqueeze(0), labels=targets[1].unsqueeze(0), return_dict=True)
                    outputs_2 = llava_model(input_ids=input_ids[2].unsqueeze(0), labels=targets[2].unsqueeze(0), return_dict=True)
                elif model_name == "Idefics2":
                    outputs_0 = llava_model(input_ids=input_ids[0].unsqueeze(0), labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_1 = llava_model(input_ids=input_ids[1].unsqueeze(0), labels=targets[1].unsqueeze(0), return_dict=True)
                    outputs_2 = llava_model(input_ids=input_ids[2].unsqueeze(0), labels=targets[2].unsqueeze(0), return_dict=True)
                elif model_name == "llava_next":
                    outputs_0 = llava_model(input_ids=input_ids[0].unsqueeze(0), images=o_input_ids[0]['pixel_values'].to(torch.float32), image_sizes=image_sizes[0], labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_1 = llava_model(input_ids=input_ids[1].unsqueeze(0), images=o_input_ids[1]['pixel_values'].to(torch.float32), image_sizes=image_sizes[1], labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_2 = llava_model(input_ids=input_ids[2].unsqueeze(0), images=o_input_ids[2]['pixel_values'].to(torch.float32), image_sizes=image_sizes[2], labels=targets[0].unsqueeze(0), return_dict=True)
                elif model_name == "llava":
                    outputs_0 = llava_model(input_ids=input_ids[0].unsqueeze(0), images=o_input_ids[0]['pixel_values'].to(torch.float32), image_sizes=image_sizes[0], labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_1 = llava_model(input_ids=input_ids[1].unsqueeze(0), images=o_input_ids[1]['pixel_values'].to(torch.float32), image_sizes=image_sizes[1], labels=targets[0].unsqueeze(0), return_dict=True)
                    outputs_2 = llava_model(input_ids=input_ids[2].unsqueeze(0), images=o_input_ids[2]['pixel_values'].to(torch.float32), image_sizes=image_sizes[2], labels=targets[0].unsqueeze(0), return_dict=True)

                # Scale and stabilize logits
                out_labels=outputs_0['labels']
                if model_name == "llava":
                    logits_1 = outputs_0['logits'][0][out_labels[0]!=-100][1:]
                    logits_2 = outputs_1['logits'][0][out_labels[0]!=-100][1:]
                    logits_3 = outputs_2['logits'][0][out_labels[0]!=-100][1:]
                elif model_name == "llava_next":
                    logits_1 = outputs_0['logits'][0][out_labels[0]!=-100][1:]
                    logits_2 = outputs_1['logits'][0][out_labels[0]!=-100][1:]
                    logits_3 = outputs_2['logits'][0][out_labels[0]!=-100][1:]
                elif model_name == "qwen-vl":
                    logits_1 = outputs_0['logits'][0][out_labels[0]!=-100]
                    logits_2 = outputs_1['logits'][0][out_labels[0]!=-100]
                    logits_3 = outputs_2['logits'][0][out_labels[0]!=-100]
                elif model_name == "InternVL3":
                    logits_1 = outputs_0['logits'][0][out_labels[0]!=-100]
                    logits_2 = outputs_1['logits'][0][out_labels[0]!=-100]
                    logits_3 = outputs_2['logits'][0][out_labels[0]!=-100]
                elif model_name == "Idefics2":
                    logits_1 = outputs_0['logits'][0][out_labels[0]!=-100][1:]
                    logits_2 = outputs_1['logits'][0][out_labels[0]!=-100][1:]
                    logits_3 = outputs_2['logits'][0][out_labels[0]!=-100][1:]
                np.save(output_folder_path + file_name[0][:-2] + '_ori_0.npy', output_text_0)
                np.save(output_folder_path + file_name[0][:-2] + '_ori_1.npy', output_text_1)
                np.save(output_folder_path + file_name[0][:-2] + '_ori_2.npy', output_text_2)
                logits_1 = torch.mean(logits_1, 0).unsqueeze(0)
                logits_2 = torch.mean(logits_2, 0).unsqueeze(0)
                logits_3 = torch.mean(logits_3, 0).unsqueeze(0)
                loss_entorpy = (entropy_loss(logits_1, logits_2) + entropy_loss(logits_1, logits_3) + entropy_loss(logits_2, logits_3))/3.
                
                loss_0 = outputs_0.loss
                loss_1 = outputs_1.loss
                loss_2 = outputs_2.loss
                loss_pseudo_label = (loss_0 + loss_1 + loss_2) / 3.0

                # Print statistics before loss computation
                loss = alpha * loss_entorpy + beta * loss_pseudo_label
                print(f"Pseudo Label Loss value: {loss_pseudo_label.item():.4f}")
                print(f"Entropy Loss value: {loss_entorpy.item():.4f}")
                print(f"Loss value: {loss.item():.4f}")
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Warning: Loss is NaN or Inf, skipping batch")
                    continue
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                output_text_0 = get_text_outputs(llava_model, processor, o_input_ids[0], args)
                output_text_1 = get_text_outputs(llava_model, processor, o_input_ids[1], args)
                output_text_2 = get_text_outputs(llava_model, processor, o_input_ids[2], args)
                if model_name == "qwen-vl":
                    output_text_0 = output_text_0.split('assistant\n')[-1]
                    output_text_1 = output_text_1.split('assistant\n')[-1]
                    output_text_2 = output_text_2.split('assistant\n')[-1]
                elif model_name == "InternVL3":
                    output_text_0 = output_text_0.split('assistant\n')[-1]
                    output_text_1 = output_text_1.split('assistant\n')[-1]
                    output_text_2 = output_text_2.split('assistant\n')[-1]
                if model_name == "Idefics2":
                    output_text_0 = output_text_0.split('Assistant: ')[-1]
                    output_text_1 = output_text_1.split('Assistant: ')[-1]
                    output_text_2 = output_text_2.split('Assistant: ')[-1]
                elif model_name == "llava":
                    output_text_0 = output_text_0.split('ASSISTANT: ')[1]
                    output_text_1 = output_text_1.split('ASSISTANT: ')[1]
                    output_text_2 = output_text_2.split('ASSISTANT: ')[1]
                elif model_name == "llava_next":
                    output_text_0 = output_text_0.split('[/INST]')[1][1:-1]
                    output_text_1 = output_text_1.split('[/INST]')[1][1:-1]
                    output_text_2 = output_text_2.split('[/INST]')[1][1:-1]
                print('answer_1: ', output_text_0, 'answer_2: ', output_text_1, 'answer_3: ', output_text_2)
                np.save(output_folder_path + file_name[0][:-2] + '_step_'+str(step)+'_0.npy', output_text_0)
                np.save(output_folder_path + file_name[0][:-2] + '_step_'+str(step)+'_1.npy', output_text_1)
                np.save(output_folder_path + file_name[0][:-2] + '_step_'+str(step)+'_2.npy', output_text_2)
                final_output_answers.append([output_text_0, output_text_1, output_text_2])

        if task == 'restyle':        
            output_text_0, output_text_1, output_text_2, output_text_4, _ = find_best_answers(final_output_answers, [pseudo_label] * update_steps)
            np.save(output_folder_path + file_name[0] + '_0.npy', output_text_0)
            np.save(output_folder_path + file_name[0] + '_1.npy', output_text_1)
            np.save(output_folder_path + file_name[0] + '_2.npy', output_text_2)
            np.save(output_folder_path + file_name[0] + '_3.npy', output_text_4)
        else:
            output_text_0, output_text_1, output_text_2, _ = find_best_answers(final_output_answers, [pseudo_label] * update_steps)
            np.save(output_folder_path + file_name[0][:-2] + '_0.npy', output_text_0)
            np.save(output_folder_path + file_name[0][:-2] + '_1.npy', output_text_1)
            np.save(output_folder_path + file_name[0][:-2] + '_2.npy', output_text_2)


if __name__ == '__main__':
    main()

