"""BERT finetuning runner."""
# from __future__ import absolute_import, division, print_function


import numpy as np
import torch
import random
import argparse
import csv
import json
from collections import defaultdict
from scipy.special import softmax
from scipy import stats
from sklearn.metrics import accuracy_score

import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from tqdm import tqdm
from transformers import  AutoTokenizer, AutoModel
from transformers.optimization import AdamW

from transformers import RobertaTokenizer
from transformers import RobertaModel   #RobertaForSequenceClassification
from sklearn.metrics import pairwise
from sentence_transformers import SentenceTransformer


class RobertaForTopKEntailment(nn.Module):
    def __init__(self, K, len_tokenizer):
        super(RobertaForTopKEntailment, self).__init__()
        self.K = K

        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        self.roberta_single.resize_token_embeddings(len_tokenizer)

        self.mlp = nn.Sequential(nn.Linear(bert_hidden_dim, bert_hidden_dim),
                                 nn.ReLU(),
                                 nn.LayerNorm(bert_hidden_dim),
                                 nn.Linear(bert_hidden_dim, 1))

    def forward(self, input_ids, input_mask, segment_ids, embedding_method):
        outputs_single = self.roberta_single(input_ids, input_mask, None)
        hidden_states = outputs_single[0] #torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)
        slice_position = self.get_label_index(segment_ids)
        # top_K_label_hidden_states shape: (batch_size, K, hidden)
        top_K_label_hidden_states = self.get_label_embedding(hidden_states, slice_position, embedding_method)

        score_single = self.mlp(top_K_label_hidden_states) #(batch, K, 2) # top_K_label_hidden_states
        return score_single 


    def get_label_index(self, segment_ids):
        """ 
        for each intent-top_K_label pair, 
        get the start and end postions for each label's tokens in the sequence.
        This will help compute mean embeddings of a label 

        segment_ids: used to slice each label in the whole concat sequence
        """
        slice_position = []
        for i_th_label in np.arange(self.K):
            row, column = np.where(segment_ids.cpu() == i_th_label)
            for j_th_batch in np.arange(segment_ids.size()[0]):
                position_in_column = np.where(row == j_th_batch)
                start = np.min(position_in_column)
                end = np.max(position_in_column)
                
                i_th_label_start = column[start+1]
                i_th_label_end = column[end]

                slice_position.append((i_th_label_start, i_th_label_end))

        slice_position = np.array(slice_position)
        slice_position = slice_position.reshape(self.K, segment_ids.size()[0], 2)
        slice_position = np.transpose(slice_position, (1, 0, 2))

        return slice_position
    
    def get_label_embedding(self, hidden_states, slice_position, flag):
        """ 
        For all the top-K labels, 
        use their token embeddings' mean/sum to represent them 
        """
        top_K_label_hidden_states = torch.zeros((1, self.K, hidden_states.size()[2]))
        for i_th_batch, slices in enumerate(slice_position):
            sliced_embedding = []
            for j_th_slice in slices:
                # print(hidden_states[i_th_batch][j_th_slice[0]: j_th_slice[1], :])
                label_embeddings = hidden_states[i_th_batch][j_th_slice[0]: j_th_slice[1], :]
                if flag == 'mean':
                    label_embedding = torch.mean(label_embeddings, 0)
                if flag == 'sum':
                    label_embedding = torch.sum(label_embeddings, 0)

                sliced_embedding.append(label_embedding)
                
            top_K_label_hidden_states = torch.cat((top_K_label_hidden_states.to('cuda'), torch.stack(sliced_embedding).unsqueeze(0)), 0)
        return top_K_label_hidden_states[1:]


class InputFeatures():
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, class_segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.class_segment_ids = class_segment_ids


def load_all_data(data_path):
    example_list = []
    with open(data_path, newline='') as f:
        for row in f:
            utterance = row.split('\t')[1].strip()
            label = row.split('\t')[0].strip()
            positive_example = {'utterance': utterance, 'class': label}
            example_list.append(positive_example)
    return example_list

def load_categories(json_fname):
    f = open(json_fname, 'r')
    cat_list = json.load(f)
    return cat_list

def convert_examples_to_features(flag, examples, top_K_candidates, ground_truth_indicator, all_class_list,  max_seq_length,
                                 tokenizer, 
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=-2,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    class_map = {label : i for i, label in enumerate(all_class_list)}

    max_length_in_data = 0

    features = []
    
    for (ex_index, example) in enumerate(tqdm(examples, desc='constructing sequence')):
        
        tokens = [cls_token]
        token_intent = tokenizer.tokenize(example)
        tokens += token_intent

        segment_id_indicator = -1
        segment_ids = [segment_id_indicator] * (len(tokens) + 1)
        """ 
        class_segment_ids indicates a label's real id according to the class map
        for all tokens of a same label, their corresponding class_segment_ids are the same
        This is to help produce the prediction labels at inference stage
        """
        class_segment_ids = [-1] * (len(tokens) + 1)

        for candidate in top_K_candidates[ex_index]:
            segment_id_indicator += 1
            class_ids_indicator = class_map[candidate]
            tokens += [sep_token] * 2
            token_candidate = tokenizer.tokenize(candidate)
            tokens += token_candidate
            segment_ids += [segment_id_indicator] * (len(token_candidate) + 2)
            class_segment_ids += [class_ids_indicator] * (len(token_candidate) + 2)

        tokens += [sep_token]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        max_length_in_data = max(max_length_in_data, len(input_ids))

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            class_segment_ids = ([pad_token_segment_id] * padding_length) + class_segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            class_segment_ids = class_segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(class_segment_ids) == max_seq_length

        if flag == 'train':
            label_id = ground_truth_indicator[ex_index]
        else:
            label_id = -1
    
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              class_segment_ids=class_segment_ids))
    
    print('max_length_in_data:', max_length_in_data)

    return features

def examples_to_features(flag, source_examples, top_K_candidates, ground_truth_indicator, all_class_list, args, tokenizer, batch_size, dataloader_mode='sequential'):
    source_features = convert_examples_to_features(flag,
        source_examples, top_K_candidates, ground_truth_indicator, all_class_list, args.max_seq_length, tokenizer, 
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=-2)#4 if args.model_type in ['xlnet'] else 0,)

    all_input_ids = torch.tensor(np.array([f.input_ids for f in source_features]), dtype=torch.long)
    all_input_mask = torch.tensor(np.array([f.input_mask for f in source_features]), dtype=torch.long)
    all_segment_ids = torch.tensor(np.array([f.segment_ids for f in source_features]), dtype=torch.long)
    all_label_ids = torch.tensor(np.array([f.label_id for f in source_features]), dtype=torch.long)
    all_class_segment_ids = torch.tensor(np.array([f.class_segment_ids for f in source_features]), dtype=torch.long)

    data_tensor = TensorDataset(all_input_ids, all_input_mask,
                             all_segment_ids, all_label_ids, 
                             all_class_segment_ids)
    
    if dataloader_mode=='sequential':
        sampler = SequentialSampler(data_tensor)
    else:
        sampler = RandomSampler(data_tensor)
    dataloader = DataLoader(data_tensor, sampler=sampler, batch_size=batch_size)

    return dataloader


def get_top_K_candidates(similarity_model, examples_dict, all_class_list,  device, args, flag):

    K = int(args.K)
    H = args.H
    T = args.T

    ground_truth_indicator = []

    def embedding_similarity(intent_embeddings, class_name_embeddings):
        similarity = pairwise.cosine_similarity(intent_embeddings, class_name_embeddings)
        return similarity
        
    def sentence_bert_similarity(similarity_model, utterance_list, class_name_list, device):
        similarity_model.to(device)
        with torch.no_grad():
            class_name_embeddings = similarity_model.encode(class_name_list)
            intent_embeddings = similarity_model.encode(utterance_list)
        similarity = embedding_similarity(torch.tensor(intent_embeddings), torch.tensor(class_name_embeddings))
        return similarity

    def augment_data(data, K, H, T, flag):
        """ 
        K is the K in top-K
        H means how many duplicates when ground truth in each k_th position
        Duplicate each piece of training data to K*H
        """
        data =  np.array(data)
        dim = len(data.shape)
        
        if flag == 'train':
            """ make sure each position has a positive class """
            N = K * H
            """ shuffle N times """
            # N = H
        elif flag == 'test' or flag == 'dev':
            N = T

        if dim == 1:
            augmented_data = np.vstack([data]*N)
            augmented_data = np.reshape(augmented_data, len(data) * N, order='F')
        if dim == 2:
            augmented_data = np.vstack([[data]]*N)
            augmented_data = np.reshape(augmented_data, (len(data) * N, -1), order='F')
        return augmented_data

    utterance_list = [i['utterance'] for i in examples_dict]
    truth_class_list = [' '.join(i['class'].split('_')) for i in examples_dict]
    class_name_list = [i for i in all_class_list]

    class_similarity = sentence_bert_similarity(similarity_model, utterance_list, class_name_list, device)
    top_K_indice = [np.argsort(i)[-K:] for i in class_similarity]
    top_K_candidates = [np.array(class_name_list)[i] for i in top_K_indice]

    recall = compute_recall(utterance_list, truth_class_list, top_K_candidates)
    print('!!!! Checking top-K-recall: [' + str(flag) + ']!!!!!:', recall)

    if flag == 'train':
        """ if there is a ground truth in top-K candidates, if not, replace the
            the class with smallest similarity with the ground truth """
        miss = 0
        for index, truth in enumerate(truth_class_list):
            if truth in top_K_candidates[index]:
                continue
            else:
                miss += 1
                top_K_candidates[index][0] = truth
                # print('miss index:', index)
        print('miss:', miss)

        utterance_list = augment_data(utterance_list, K, H, T, 'train')
        truth_class_list = augment_data(truth_class_list, K, H, T, 'train')
        top_K_candidates = augment_data(top_K_candidates, K, H, T, 'train')

        """ make sure each position has a positive class """ 
        """ return a ground truth index indicator in candidates """
        for index, truth in enumerate(truth_class_list):
            ground_truth_index = np.where(top_K_candidates[index] == truth)
            candidates_without_truth = np.delete(top_K_candidates[index], ground_truth_index)
            np.random.shuffle(candidates_without_truth)
            truth_position = index % (K*H) // H
            augmented_candidate = np.insert(candidates_without_truth, truth_position,truth)
            top_K_candidates[index] = augmented_candidate 
            indicator = np.isin(np.asarray(top_K_candidates[index]), np.asarray(truth)).astype(int)
            ground_truth_indicator.append(indicator)

    else:
        utterance_list = augment_data(utterance_list, K, H, T, flag)
        truth_class_list = augment_data(truth_class_list, K, H, T, flag)
        top_K_candidates = augment_data(top_K_candidates, K, H, T, flag)   

        for index, truth in enumerate(truth_class_list):
            temp_candidates = top_K_candidates[index]
            np.random.shuffle(temp_candidates)
            top_K_candidates[index] = temp_candidates
   
    return top_K_candidates, ground_truth_indicator, truth_class_list, utterance_list

def compute_recall(utterance_list, class_list, class_candidates):
    total = len(utterance_list)
    hit = 0
    for index, utterance in enumerate(utterance_list):
        top_K_candidates = class_candidates[index]
        true_class = class_list[index]
        if true_class in top_K_candidates:
            hit += 1
    return hit/total


def metric(preds, top_K_candidates, truth_label_list, class_map, T):
    pred_label_index = np.argmax(preds, axis=1) 
    pred_results = np.array(top_K_candidates)[np.arange(len(top_K_candidates)), np.array(pred_label_index)]
    
    pred_results = [class_map[i] for i in pred_results]
    test_ground_truth_class_id = [class_map[i] for i in truth_label_list]

    acc = accuracy_score(test_ground_truth_class_id, pred_results)
    
    pred_results = np.array(pred_results).reshape(-1, T)
    pred_results = stats.mode(pred_results, axis=1).mode.flatten()
    test_ground_truth_class_id = np.array(test_ground_truth_class_id).reshape(-1, T)[:, 0]
    acc = accuracy_score(test_ground_truth_class_id, pred_results)
    acc = f"{acc:.2%}"
    return acc
   

def evaluate(model, dev_dataloader, test_dataloader, device, dev_top_K_candidates, test_top_K_candidates, class_map, dev_truth_label_list, test_truth_label_list, args):
    """ evaluate """
    model.eval()    
    len_test = len(test_truth_label_list)
    len_dev = len(dev_truth_label_list)
    """ get pred for dev data """
    dev_preds = []
    for _, batch in enumerate(tqdm(dev_dataloader, desc="load dev data")):
        input_ids, input_mask, segment_ids, label_ids, class_segment_ids = batch
        
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, input_mask, segment_ids, args.embedding_method)

        if len(dev_preds) == 0:
            dev_preds.append(logits.detach().cpu().numpy())
        else:
            dev_preds[0] = np.append(dev_preds[0], logits.detach().cpu().numpy(), axis=0)

    dev_preds = dev_preds[0].reshape(dev_preds[0].shape[0], -1)
    dev_preds = softmax(dev_preds,axis=1)

    dev_performance = metric(dev_preds, dev_top_K_candidates, dev_truth_label_list, class_map, args.T)

    """ get pred for test data """
    test_preds = []
    for _, batch in enumerate(tqdm(test_dataloader, desc="load test data")):
        input_ids, input_mask, segment_ids, label_ids, class_segment_ids = batch
        
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, input_mask, segment_ids, args.embedding_method)

        if len(test_preds) == 0:
            test_preds.append(logits.detach().cpu().numpy())
        else:
            test_preds[0] = np.append(test_preds[0], logits.detach().cpu().numpy(), axis=0)
   
    test_preds = test_preds[0].reshape(test_preds[0].shape[0], -1)
    test_preds = softmax(test_preds,axis=1)

    test_performance = metric(test_preds, test_top_K_candidates, test_truth_label_list, class_map, args.T)

    print('-------------------')
    return dev_performance, test_performance

def main(args_train_batch_size, args_test_batch_size, args_num_train_epochs, args_learning_rate, args_ENABLE_WANDB, args_K, args_embedding_method, args_seed, args_eval_each_epoch, args_T, args_max_seq_length, args_train_file, args_save_epochs, args_H):

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_batch_size",
                        default=args_train_batch_size,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--test_batch_size",
                        default=args_test_batch_size,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=args_learning_rate,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=args_num_train_epochs,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval_each_epoch",
                        default=args_eval_each_epoch,
                        action='store_true',
                        help="For each entity, sample its top similar negative labels to construct the negative pairs. If set to False, do random sample")
    parser.add_argument("--save_epochs",
                        default=args_save_epochs,
                        type=int,
                        help="Save checkpoint every X epochs of training")
    parser.add_argument("--embedding_method",
                        default=args_embedding_method,
                        type=str,
                        help="Use mean or sum to get embeddings")
    parser.add_argument("--K",
                        default=args_K,
                        type=int,
                        help="Total number of top candidates selected")  
    parser.add_argument("--H",
                        default=args_H,
                        type=int,
                        help="Total number of top candidates selected")  
    parser.add_argument("--T",
                        default=args_T,
                        type=int,
                        help="The number of augmentation for each piece of data")  
    parser.add_argument("--ENABLE_WANDB",
                        default=args_ENABLE_WANDB,
                        action='store_true',
                        help="Use wandb or not.")
    parser.add_argument('--seed',
                        type=int,
                        default=args_seed,
                        help="random seed for initialization")
    parser.add_argument('--result_name',
                        type=str,
                        default=args_result_name)
    parser.add_argument("--max_seq_length",
                        default=args_max_seq_length,
                        type=int,
                        help="max_seq_length")
    parser.add_argument("--train_file",
                        default=args_train_file,
                        type=str,
                        help="the name of train file")

    parser.add_argument('-f')
 
    args = parser.parse_args()   

    train_path = root_dir + 'data/banking_data/sampled_data/' + str(args.train_file) + '.txt'
    device = torch.device("cuda")
    """ set random seed """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    """ load data  """
    all_class_list = load_categories(category_path)
    all_class_list = [' '.join(i.split('_')) for i in all_class_list]

    train_example_list = load_all_data(train_path)
    dev_example_list = load_all_data(dev_path)
    test_example_list = load_all_data(test_path)
    class_map = {' '.join(label.split('_')) : i for i, label in enumerate(all_class_list)}

    """ load top-k Entailment model """
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=True)


    model = RobertaForTopKEntailment(args.K, len(tokenizer))
    model.to(device)


    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)


    """ Top-K selection """
    """ load top-K selection model """
    similarity_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    train_top_K_candidates, train_ground_truth_indicator, _, train_utterance_list = get_top_K_candidates(similarity_model, train_example_list, all_class_list,  device, args, 'train')


    dev_top_K_candidates, _, dev_truth_label_list, dev_utterance_list = get_top_K_candidates(similarity_model, dev_example_list, all_class_list,  device, args, 'dev')
    
    test_top_K_candidates, _, test_truth_label_list, test_utterance_list = get_top_K_candidates(similarity_model, test_example_list, all_class_list,  device, args, 'test')

    test_ground_truth_class_id = [class_map[i] for i  in test_truth_label_list]
    dev_ground_truth_class_id = [class_map[i] for i in dev_truth_label_list]

    print('Top-K selection')
    """ ------------------- """
    
    train_dataloader = examples_to_features('train', train_utterance_list, train_top_K_candidates, train_ground_truth_indicator, all_class_list, args, tokenizer, args.train_batch_size, dataloader_mode='random')

    dev_dataloader = examples_to_features('dev', dev_utterance_list, dev_top_K_candidates, dev_ground_truth_class_id, all_class_list, args, tokenizer, args.test_batch_size, dataloader_mode='sequential')

    test_dataloader = examples_to_features('test', test_utterance_list, test_top_K_candidates, test_ground_truth_class_id, all_class_list, args, tokenizer, args.test_batch_size, dataloader_mode='sequential')

    """ training """
    performence_each_epoch = []
    for epoch in range(args.num_train_epochs):
        for _, batch in enumerate(tqdm(train_dataloader, desc='train|epoch'+str(epoch))):
            model.train()
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids, label_ids, class_segment_ids= batch

            logits = model(input_ids, input_mask, segment_ids, args.embedding_method)

            bcsz= input_ids.shape[0]

            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(bcsz, -1), label_ids.to(device).float())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if args.eval_each_epoch and (epoch+1) % args.save_epochs == 0:
            dev_performance, test_performance = evaluate(model, dev_dataloader, test_dataloader, device, dev_top_K_candidates, test_top_K_candidates, class_map, dev_truth_label_list, test_truth_label_list, args)
            print(args.train_file, ' dev_performance:', dev_performance)
            print(args.train_file, ' test_performance:', test_performance)
            print('-------------------')
            
            training_details = f'ep{epoch}_{args.train_file}_'f'K{args.K}_H_{args.H}_SEED{args.seed}'

            performence_each_epoch.append((dev_performance, test_performance, training_details))


    final_test_performance = sorted(performence_each_epoch, key=lambda x: -float(x[0].replace('%', 'e-2')))[0][1]
    final_dev_performance = sorted(performence_each_epoch, key=lambda x: -float(x[0].replace('%', 'e-2')))[0][0]
    final_model = sorted(performence_each_epoch, key=lambda x: -float(x[0].replace('%', 'e-2')))[0][2]

if __name__ == "__main__":
    args_train_batch_size = 8 
    args_test_batch_size = 128  
    args_num_train_epochs = 2
    args_learning_rate = 5e-6
    args_ENABLE_WANDB = False
    args_K =25
    args_H = 1
    args_T = 1
    args_embedding_method = 'mean' # 'sum' or 'mean'
    args_seed = 36
    args_result_name = ''
    args_eval_each_epoch = True
    args_max_seq_length = 400 # 320 if K is 35.   # K = 60 max:442    
    args_train_file = 'five_shot_0'
    args_save_epochs = 1

    """ LOCAL  """
    root_dir = '../'

    """ --------- """

    bert_hidden_dim = 1024
    pretrain_model_dir = 'roberta-large' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'

    dev_path = root_dir + 'data/banking_data/sampled_data/dev.txt'
    test_path = root_dir + 'data/banking_data/sampled_data/test.txt'
    
    category_path = root_dir + 'data/banking_data/categories.json'

    main(args_train_batch_size, args_test_batch_size, args_num_train_epochs, args_learning_rate, args_ENABLE_WANDB, args_K, args_embedding_method, args_seed, args_eval_each_epoch, args_T, args_max_seq_length, args_train_file,args_save_epochs, args_H)