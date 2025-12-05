import pdb
import os, sys, random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
import copy

def load_image(image_file):
    return Image.open(image_file).convert("RGB")

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

class RephraseDataset(data.Dataset):
    def __init__(self, folder_root, image_path, text_path, model, processor, evaluate, arg_index, model_name):

        self.image_path = image_path
        self.text_path = text_path
        data_all = []
        for ele in os.listdir(text_path):
            ele = ele.replace('.txt', '')
            if ele[:-2] not in data_all:
                data_all.append(ele[:-2])
        self.data = data_all
        self.okvqa_gt = np.load(folder_root + 'answers_okvqa.npy', allow_pickle=True).item()
        self.infovqa_gt = np.load(folder_root + 'answers_infovqa.npy', allow_pickle=True).item()
        self.infovqa_path = self.image_path
        self.okvqa_path = self.image_path
        self.model_name = model_name
        self.model = model
        self.processor = processor
        self.pad_len = 10
        self.evaluate = evaluate

        self.index = arg_index

    def __getitem__(self, index):

        index = self.index
        data = self.data[index]
        data = data + '_0'
        output_input_ids = []
        output_targets = []
        original_inputs = []
        for ddd in range(3):
            select_number = ddd
            question = 'Please answer the question with a very-short answer. ' + open(self.text_path + data[:-2] + '_' +str(select_number) + '.txt').read()
            qs = "Question: "+question+" Answer:"

            try:
                image_files = self.infovqa_path + data.split('_')[0]+'.jpeg'
                answer = self.infovqa_gt[data[:-2]][0]
            except:
                img_file_name = 'COCO_train2014_%012d.jpg' %int(data.split('_')[0])
                image_files = self.okvqa_path + img_file_name
                answer = self.okvqa_gt[data[:-2]][0]
            image = Image.open(image_files)
            image_sizes = [image.size]

            if self.model_name == "llava":
                conversation = [
                    {
                        "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": qs},
                    ],
                },
            ]
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(image, prompt, return_tensors="pt").to("cuda:0")
            elif self.model_name == "llava_next":
                conversation = [
                    {
                        "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": qs},
                    ],
                },
            ]
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(image, prompt, return_tensors="pt").to("cuda:0")
            elif self.model_name == "qwen-vl":
                conversation = [
                        {
                            "role":"user",
                            "content":[
                                {
                                    "type":"image",
                                    "path": image_files
                                },
                                {
                                    "type":"text",
                                    "text":qs
                                }
                            ]
                        }
                    ]
                inputs = self.processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt")
                inputs = inputs.to("cuda:0")
            elif self.model_name == "InternVL3":
                conversation = [
                        {
                            "role":"user",
                            "content":[
                                {
                                    "type":"image",
                                    "path": image_files
                                },
                                {
                                    "type":"text",
                                    "text":qs
                                }
                            ]
                        }
                    ]
                inputs = self.processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt")
                inputs = inputs.to("cuda:0")
            elif self.model_name == "Idefics2":
                conversation = [
                        {
                            "role":"user",
                            "content":[
                                {
                                    "type":"image",
                                    "path": image_files
                                },
                                {
                                    "type":"text",
                                    "text":qs
                                }
                            ]
                        }
                    ]
                inputs = self.processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt")
                inputs = inputs.to("cuda:0")

            original_inputs.append(inputs)
            input_ids = torch.nn.functional.pad(inputs['input_ids'], (0, self.pad_len), value=0)
            output_input_ids.append(input_ids)

            answer = answer + self.processor.tokenizer.eos_token
            target_ids = self.processor.tokenizer(answer, add_special_tokens=True,max_length=self.pad_len, truncation=True).input_ids
            target_ids = torch.tensor(target_ids, dtype=torch.long)
            target_ids = target_ids.unsqueeze(0).cuda()
            target_ids = target_ids.cuda()
            decode_attention = target_ids.masked_fill(target_ids!=self.processor.tokenizer.pad_token_id, 1)
            targets = torch.full_like(input_ids, -100)
            if target_ids.shape[1] >= self.pad_len:
                targets[:,-self.pad_len:] = target_ids[:,:self.pad_len]
            else:
                targets[:,-self.pad_len:-self.pad_len+target_ids.shape[1]] = target_ids
            output_targets.append(targets)

            file_name = data

        input_ids_max_len = max(output_input_ids[0].shape[1], output_input_ids[1].shape[1], output_input_ids[2].shape[1])
        tt0 = torch.nn.functional.pad(output_input_ids[0], (0, input_ids_max_len - output_input_ids[0].size(1)), value=0)
        tt1 = torch.nn.functional.pad(output_input_ids[1], (0, input_ids_max_len - output_input_ids[1].size(1)), value=0)
        tt2 = torch.nn.functional.pad(output_input_ids[2], (0, input_ids_max_len - output_input_ids[2].size(1)), value=0)
        output_input_ids = torch.vstack([tt0, tt1, tt2])

        targets_max_len = max(output_targets[0].shape[1], output_targets[1].shape[1], output_targets[2].shape[1])
        rr0 = torch.nn.functional.pad(output_targets[0], (0, targets_max_len - output_targets[0].size(1)), value=-100)
        rr1 = torch.nn.functional.pad(output_targets[1], (0, targets_max_len - output_targets[1].size(1)), value=-100)
        rr2 = torch.nn.functional.pad(output_targets[2], (0, targets_max_len - output_targets[2].size(1)), value=-100)
        output_targets = torch.vstack([rr0, rr1, rr2])

        return image_sizes, output_input_ids, output_targets, file_name, original_inputs

    def __len__(self):
        return len(self.data)

class MaskingDataset(data.Dataset):
    def __init__(self, folder_root, image_path, text_path, model, processor, evaluate, arg_index, model_name):

        self.image_path = image_path
        self.text_path = text_path
        self.data_tmp = os.listdir(self.image_path)
        
        data_all = []
        for ele in self.data_tmp:
            ele = ele.replace('.jpg', '')
            if ele[:-2] not in data_all:
                data_all.append(ele[:-2])
        self.data = data_all
        print('len(self.data): ', len(self.data))
        self.labels = np.load(folder_root + 'labels.npy', allow_pickle=True).item()
        self.categories = np.load(folder_root + 'coco_category_dic.npy', allow_pickle=True).item()

        self.model_name = model_name
        self.model = model
        self.processor = processor
        self.pad_len = 10
        self.evaluate = evaluate

        self.index = arg_index

    def __getitem__(self, index):

        index = self.index
        data = self.data[index]
        #import pdb; pdb.set_trace()
        data = data + '_0'
        output_input_ids = []
        output_targets = []
        original_inputs = []
        for ddd in range(3):
            qs = "Question: Please answer the question with a very-short answer. What kind of object is in the masked region? Answer:"

            answer = self.categories[self.labels[data[:-2]+'_0']]

            image_files = self.image_path + data[:-1] + str(ddd)+ '.jpg'
            image = Image.open(image_files)
            image_sizes = [image.size]

            if self.model_name == "llava":
                conversation = [
                    {
                      "role": "user",
                      "content": [
                            {"type": "text", "text": qs},
                            {"type": "image"},
                        ],
                    },
                ]
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(image, prompt, return_tensors="pt").to("cuda:0")
            elif self.model_name == "llava_next":
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": qs},
                        ],
                    },
                ]
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(image, prompt, return_tensors="pt").to("cuda:0")
            elif self.model_name == "qwen-vl":
                conversation = [
                        {
                            "role":"user",
                            "content":[
                                {
                                    "type":"image",
                                    "path": image_files
                                },
                                {
                                    "type":"text",
                                    "text":qs
                                }
                            ]
                        }
                    ]
                inputs = self.processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt").to("cuda:0")
            elif self.model_name == "InternVL3":
                conversation = [
                        {
                            "role":"user",
                            "content":[
                                {
                                    "type":"image",
                                    "path": image_files
                                },
                                {
                                    "type":"text",
                                    "text":qs
                                }
                            ]
                        }
                    ]
                inputs = self.processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt").to("cuda:0")
            elif self.model_name == "Idefics2":
                conversation = [
                        {
                            "role":"user",
                            "content":[
                                {
                                    "type":"image",
                                    "path": image_files
                                },
                                {
                                    "type":"text",
                                    "text":qs
                                }
                            ]
                        }
                    ]
                inputs = self.processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt")
                inputs = inputs.to("cuda:0")

            original_inputs.append(inputs)
            input_ids = torch.nn.functional.pad(inputs['input_ids'], (0, self.pad_len), value=0)
            output_input_ids.append(input_ids)

            answer = answer + self.processor.tokenizer.eos_token
            target_ids = self.processor.tokenizer(answer, add_special_tokens=True,max_length=self.pad_len, truncation=True).input_ids
            target_ids = torch.tensor(target_ids, dtype=torch.long)
            target_ids = target_ids.unsqueeze(0).cuda()
            target_ids = target_ids.cuda()
            decode_attention = target_ids.masked_fill(target_ids!=self.processor.tokenizer.pad_token_id, 1)
            targets = torch.full_like(input_ids, -100)
            if target_ids.shape[1] >= self.pad_len:
                targets[:,-self.pad_len:] = target_ids[:,:self.pad_len]
            else:
                targets[:,-self.pad_len:-self.pad_len+target_ids.shape[1]] = target_ids
            output_targets.append(targets)

            file_name = data

        input_ids_max_len = max(output_input_ids[0].shape[1], output_input_ids[1].shape[1], output_input_ids[2].shape[1])
        tt0 = torch.nn.functional.pad(output_input_ids[0], (0, input_ids_max_len - output_input_ids[0].size(1)), value=0)
        tt1 = torch.nn.functional.pad(output_input_ids[1], (0, input_ids_max_len - output_input_ids[1].size(1)), value=0)
        tt2 = torch.nn.functional.pad(output_input_ids[2], (0, input_ids_max_len - output_input_ids[2].size(1)), value=0)
        output_input_ids = torch.vstack([tt0, tt1, tt2])

        targets_max_len = max(output_targets[0].shape[1], output_targets[1].shape[1], output_targets[2].shape[1])
        rr0 = torch.nn.functional.pad(output_targets[0], (0, targets_max_len - output_targets[0].size(1)), value=-100)
        rr1 = torch.nn.functional.pad(output_targets[1], (0, targets_max_len - output_targets[1].size(1)), value=-100)
        rr2 = torch.nn.functional.pad(output_targets[2], (0, targets_max_len - output_targets[2].size(1)), value=-100)
        output_targets = torch.vstack([rr0, rr1, rr2])

        return image_sizes, output_input_ids, output_targets, file_name, original_inputs

    def __len__(self):
        return len(self.data)

class StyleDataset(data.Dataset):
    def __init__(self, folder_root, image_path, text_path, model, processor, evaluate, arg_index, model_name):
        
        self.image_path = image_path
        self.data = []
        
        for ele in os.listdir(self.image_path):
            element = ele.replace('_candy.jpg','').replace('_mosaic.jpg','').replace('_udnie.jpg','').replace('_grayscale.jpg','')
            if element not in self.data:
                self.data.append(element)
        print('len(self.data): ', len(self.data))
        self.indoor_labels = np.load(folder_root + 'indoor_labels.npy', allow_pickle=True).item()
        self.landmarks_labels = np.load(folder_root + 'landmarks_labels.npy', allow_pickle=True).item()
        

        self.model = model
        self.processor = processor
        self.model_name = model_name
        self.pad_len = 10
        self.evaluate = evaluate

        self.index = arg_index

    def __getitem__(self, index):

        index = self.index
        data = self.data[index]
        
        output_input_ids = []
        output_targets = []
        original_input_ids = []
        image_tensor_list = []
        for mmm in ['candy', 'mosaic', 'udnie','grayscale']:
            question = "Please describe the place in the image in very-short answer."
            qs = "Question: "+question+" Answer:"

            try:
                image_files = [self.image_path + data +'_'+mmm+'.jpg']
                answer = self.indoor_labels[data]
            except:
                image_files = [self.image_path + data +'_'+mmm+'.jpg']
                answer = self.landmarks_labels[data]
            image = Image.open(image_files[0])
            image_sizes = [image.size]

            if self.model_name == "llava":
                conversation = [
                    {
                        "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": qs},
                    ],
                },
            ]
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(image, prompt, return_tensors="pt").to("cuda:0")
            elif self.model_name == "llava_next":
                conversation = [
                    {
                        "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": qs},
                    ],
                },
            ]
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(image, prompt, return_tensors="pt").to("cuda:0")
            elif self.model_name == "InternVL3":
                conversation = [
                        {
                            "role":"user",
                            "content":[
                                {
                                    "type":"image",
                                    "path": image_files[0]
                                },
                                {
                                    "type":"text",
                                    "text":qs
                                }
                            ]
                        }
                    ]
                inputs = self.processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt")
                inputs = inputs.to("cuda:0")
            elif self.model_name == "qwen-vl":
                conversation = [
                        {
                            "role":"user",
                            "content":[
                                {
                                    "type":"image",
                                    "path": image_files[0]
                                },
                                {
                                    "type":"text",
                                    "text":qs
                                }
                            ]
                        }
                    ]
                inputs = self.processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt")
            elif self.model_name == "Idefics2":
                conversation = [
                        {
                            "role":"user",
                            "content":[
                                {
                                    "type":"image",
                                    "path": image_files[0]
                                },
                                {
                                    "type":"text",
                                    "text":qs
                                }
                            ]
                        }
                    ]
                inputs = self.processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt")
                inputs = inputs.to("cuda:0")

            original_input_ids.append(inputs)
            input_ids = torch.nn.functional.pad(inputs['input_ids'], (0, self.pad_len), value=0)
            output_input_ids.append(input_ids)

            answer = answer + self.processor.tokenizer.eos_token
            target_ids = self.processor.tokenizer(answer, add_special_tokens=True,max_length=self.pad_len, truncation=True).input_ids
            target_ids = torch.tensor(target_ids, dtype=torch.long)
            target_ids = target_ids.unsqueeze(0).cuda()
            target_ids = target_ids.cuda()
            decode_attention = target_ids.masked_fill(target_ids!=self.processor.tokenizer.pad_token_id, 1)
            targets = torch.full_like(input_ids, -100)
            if target_ids.shape[1] >= self.pad_len:
                targets[:,-self.pad_len:] = target_ids[:,:self.pad_len]
            else:
                targets[:,-self.pad_len:-self.pad_len+target_ids.shape[1]] = target_ids
            output_targets.append(targets)

            file_name = data
        input_ids_max_len = max(output_input_ids[0].shape[1], output_input_ids[1].shape[1], output_input_ids[2].shape[1], output_input_ids[3].shape[1])
        tt0 = torch.nn.functional.pad(output_input_ids[0], (0, input_ids_max_len - output_input_ids[0].size(1)), value=0)
        tt1 = torch.nn.functional.pad(output_input_ids[1], (0, input_ids_max_len - output_input_ids[1].size(1)), value=0)
        tt2 = torch.nn.functional.pad(output_input_ids[2], (0, input_ids_max_len - output_input_ids[2].size(1)), value=0)
        tt3 = torch.nn.functional.pad(output_input_ids[3], (0, input_ids_max_len - output_input_ids[3].size(1)), value=0)
        output_input_ids = torch.vstack([tt0, tt1, tt2, tt3])

        targets_max_len = max(output_targets[0].shape[1], output_targets[1].shape[1], output_targets[2].shape[1], output_targets[3].shape[1])
        rr0 = torch.nn.functional.pad(output_targets[0], (0, targets_max_len - output_targets[0].size(1)), value=-100)
        rr1 = torch.nn.functional.pad(output_targets[1], (0, targets_max_len - output_targets[1].size(1)), value=-100)
        rr2 = torch.nn.functional.pad(output_targets[2], (0, targets_max_len - output_targets[2].size(1)), value=-100)
        rr3 = torch.nn.functional.pad(output_targets[3], (0, targets_max_len - output_targets[3].size(1)), value=-100)
        output_targets = torch.vstack([rr0, rr1, rr2, rr3])
        
        return image_sizes, output_input_ids, output_targets, file_name, original_input_ids

    def __len__(self):
        return len(self.data)

