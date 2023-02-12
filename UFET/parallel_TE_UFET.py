"""BERT finetuning runner."""
# from __future__ import absolute_import, division, print_function

import bi_bert as db 

import numpy as np
import torch
import random
import wandb
import argparse
from scipy.special import softmax

import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from tqdm import tqdm
from transformers import  AutoTokenizer, AutoModel
from transformers.optimization import AdamW

from transformers import RobertaTokenizer
from transformers import RobertaModel   #RobertaForSequenceClassification


class RobertaForTopKEntailment(nn.Module):
    def __init__(self, K, len_tokenizer):
        super(RobertaForTopKEntailment, self).__init__()
        self.K = K

        self.roberta= RobertaModel.from_pretrained(pretrain_model_dir, local_files_only = True)
        self.roberta.resize_token_embeddings(len_tokenizer)

        # if concat entity embed with typing embed, the input dim should be bert_hidden_dim * 2. Otherwise, bert_hidden_dim
        self.mlp = nn.Sequential(nn.Linear(bert_hidden_dim, bert_hidden_dim),
                                 nn.ReLU(),
                                 nn.LayerNorm(bert_hidden_dim),
                                 nn.Linear(bert_hidden_dim, 1))

    def forward(self, input_ids, input_mask, segment_ids, entity_span_index, embedding_method):
        outputs_single = self.roberta(input_ids, input_mask, None)
        hidden_states = outputs_single[0] #torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)
        slice_position = self.get_label_index(segment_ids)
        # top_K_label_hidden_states shape: (batch_size, K, hidden)
        top_K_label_hidden_states = self.get_label_embedding(hidden_states, slice_position, embedding_method)
        
        # # entity_hidden_states shape: (batch_size, hidden)
        # entity_hidden_states = self.get_entity_embedding(hidden_states, entity_span_index, embedding_method)
        # # transform entity_hidden_states to the same shape as top_K_label_hidden_states so they can be concat at the final dim
        # entity_hidden_states = torch.unsqueeze(entity_hidden_states, 1).expand(-1, self.K, -1)
        # # concat each top-K label with the mentioned entity span in the statement
        # entity_label_hidden = torch.cat((top_K_label_hidden_states, entity_hidden_states), axis = 2)

        score_single = self.mlp(top_K_label_hidden_states) #(batch, K, 2) # top_K_label_hidden_states
        return score_single 

    def get_entity_embedding(self, hidden_states, entity_span_index, flag):
        entity_embed = []
        for i_th_batch, index in enumerate(entity_span_index):
            embed = hidden_states[i_th_batch][index[0]:index[1]]
            if flag == 'mean':
                embed = torch.mean(embed, 0)
            if flag == 'sum':
                embed = torch.sum(embed, 0)
            entity_embed.append(embed)
        entity_embed = torch.stack(entity_embed)
        return entity_embed

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

    def __init__(self, input_ids, input_mask, segment_ids, label_id, class_segment_ids, entity_span_index):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.class_segment_ids = class_segment_ids
        self.entity_span_index = entity_span_index


def load_all_data(data_path, type_path):
    dataset, all_type_list, _ = db.load_data(data_path, type_path)
    example_list = []
    for data_index, data in enumerate(dataset):
        left = data['left_context_token']
        entity = data['mention_span']
        right = data['right_context_token']
        entity_typing = [' '.join(i.split('_')) for i in data['y_str']]
        entitiy_typing_vanilla = data['y_str']
        entity_typing_index = [all_type_list.index(i) for i in entity_typing]
        
        left_str = ' '.join(left).lstrip()
        right_str = ' '.join(right).lstrip()

        if len(left) == 0:
            statement = left_str + '{'+ entity + '}'+ ' ' + right_str
        elif len(right) == 1:
            statement = left_str + ' ' + '{'+ entity + '}' + right_str
        else:
            statement = left_str + ' ' + '{'+ entity + '}' + ' ' + right_str
        
        positive_example = {'statement': statement, 'typing': entity_typing, 'typing_vanilla': entitiy_typing_vanilla, 'entity': entity}
        example_list.append(positive_example)

    return example_list


def convert_examples_to_features(flag, examples, entity_span_index_roberta, top_K_candidates, ground_truth_indicator, eval_class_list,  max_seq_length,
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
    class_map = {label : i for i, label in enumerate(eval_class_list)}

    max_length_in_data = 0

    features = []
    
    for (ex_index, example) in enumerate(tqdm(examples, desc='constructing sequence')):
        entity_span_index = entity_span_index_roberta[ex_index]
        
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
                              class_segment_ids=class_segment_ids,
                              entity_span_index = entity_span_index))
    
    print('max_length_in_data:', max_length_in_data)

    return features


def examples_to_features(flag, source_examples, entity_span_index_roberta, top_K_candidates, ground_truth_indicator, eval_class_list, args, tokenizer, batch_size, dataloader_mode='sequential'):
    source_features = convert_examples_to_features(flag,
        source_examples, entity_span_index_roberta, top_K_candidates, ground_truth_indicator, eval_class_list, args.max_seq_length, tokenizer, 
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=-2)#4 if args.model_type in ['xlnet'] else 0,)

    all_input_ids = torch.tensor([f.input_ids for f in source_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in source_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in source_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in source_features], dtype=torch.long)
    all_class_segment_ids = torch.tensor([f.class_segment_ids for f in source_features], dtype=torch.long)
    all_entity_span_index = torch.tensor([f.entity_span_index for f in source_features], dtype=torch.long)


    data_tensor = TensorDataset(all_input_ids, all_input_mask,
                             all_segment_ids, all_label_ids, 
                             all_class_segment_ids, all_entity_span_index)
    
    if dataloader_mode=='sequential':
        sampler = SequentialSampler(data_tensor)
    else:
        sampler = RandomSampler(data_tensor)
    dataloader = DataLoader(data_tensor, sampler=sampler, batch_size=batch_size)


    return dataloader


def extract_entity(data_list_roberta, roberta_tokenizer):
    """ extract entity tokens from all tokens tokenized by RoBERTa  """

    # roberta <mask> token to seperate entity
    entity_sep = roberta_tokenizer.encode(roberta_tokenizer.mask_token)[1]
    input_ids = roberta_tokenizer(data_list_roberta, return_tensors='np', padding ='longest').input_ids
    # input_ids = [np.array(i) for i in input_ids]

    entity_span_index = np.where(input_ids == entity_sep)[1].reshape(-1, 2) - np.array([0, 1])

    return entity_span_index

def get_entity_embedding(last_hidden_state, entity_span_index, flag = 'mean'):
    entity_embedding_list = []
    for hidden, span in zip(last_hidden_state, entity_span_index):
        if flag == 'mean':
            entity_embedding =  torch.mean(hidden[span[0]:span[1]], 0)
            # entity_embedding =  torch.mean(torch.stack((entity_embedding, hidden[0])), 0)
        elif flag == 'sum':
            entity_embedding =  torch.sum(hidden[span[0]:span[1]], 0)
            # entity_embedding =  torch.mean(torch.stack((entity_embedding, hidden[0])), 0)
        entity_embedding_list.append(entity_embedding)
    return torch.stack(entity_embedding_list)

def get_top_K_candidates(statement_model, types_vector_path, example_list, all_type_list, device, statement_tokenizer, roberta_tokenizer, args, flag):

    # load all types embeddings
    _, _, all_types_embedding_loaded = db.load_type_vector(types_vector_path)
    
    data_list = np.array([[e['statement'], '-1'] for e in example_list]) # '-1' is meaningless. It is like a placeholder so data_list can be formatted to the input of db.examples_to_features()
    data_list_roberta = [i[0].replace('{', '<mask>').replace('}', '<mask>') for i in data_list]  # sep entity with <mask> tokens, so roberta can extract the entity easily

    entity_span_index_roberta = extract_entity(data_list_roberta, roberta_tokenizer)

    statement_list = [e['statement'].replace('{', '<ent><ent> ').replace('}', ' <ent><ent>') for e in example_list]
    truth_label_list = [e['typing'] for e in example_list]
    entity_list = [e['entity'] for e in example_list]

    statement_dataloader = db.examples_to_features(data_list, statement_tokenizer, len(data_list), 'sequential')

    ground_truth_indicator = []

    # Get test entity embeddings
    for _, batch in enumerate(tqdm(statement_dataloader, desc='Getting top-K')):
        batch = tuple(t.to(device) for t in batch)
        # Get input_ids(tokens) for statements
        statement_input_ids, statement_input_mask, entity_span_index, statement_index = batch

        # Get embeddings for entity span in each statement
        with torch.no_grad():
            statement_outputs = statement_model(statement_input_ids, statement_input_mask)
        # the method of get_entity_embedding must be consistent with the pretrained top-k model. If the pretrained top-k model concat ebeb. of entity with whole sentence, in here it must be same. VICE VERSA.
        entity_embedding_list = get_entity_embedding(statement_outputs[0], entity_span_index, args.embedding_method)

    similarity = db.embedding_similarity(entity_embedding_list, all_types_embedding_loaded)
    top_K_indice = [np.argsort(i)[-args.K:] for i in similarity]
    top_K_candidates = [np.array(all_type_list)[i] for i in top_K_indice]

    # compute recall Note: if recall is not consistent with the pretrained model, consider if '_' in typing is reserved AND check get_entity_embedding method. 
    recall_list = db.compute_recall(top_K_candidates, truth_label_list)
    avg_recall = sum(recall_list)/len(recall_list)
    print('!!!! Checking top-K-recall: [' + str(flag) + ']!!!!!:', avg_recall)

    def duplicate_data(data, N):
        """ 
        Duplicate each piece of training data N times

        Data needs to be duplicated:
        top_K_candidates
        statement_list
        truth_label_list
        """
        data =  np.array(data, dtype=object)
        augmented_data = np.repeat(data, np.array([N for i in range(len(data))]), axis=0)
        return augmented_data

    if flag == 'train':
        """ if all ground truth labels are in top-K candidates, if not, replace the
            the class with smallest similarity with the ground truth """
        # miss = 0
        # for index, truth in enumerate(truth_label_list):
        #     top_K_set = set(top_K_candidates[index])
        #     truth_set = set(truth)
        #     if len(truth) == len(top_K_set & truth_set):
        #         continue
        #     else:
        #         miss += 1
        #         missed_label = truth_set - top_K_set
        #         replace_index = 0
        #         for label in missed_label:
        #             while (top_K_candidates[index][replace_index] in truth_set):
        #                 # in case that the position will be replaced is already a true label
        #                 replace_index += 1
        #             top_K_candidates[index][replace_index]  = label
        #             replace_index += 1
        #         # print('miss index:', index)      
        # print(miss)

        top_K_candidates = duplicate_data(top_K_candidates, args.N)
        statement_list = duplicate_data(statement_list, args.N)
        truth_label_list = duplicate_data(truth_label_list, args.N)
        entity_span_index_roberta = duplicate_data(entity_span_index_roberta, args.N)

        """ shuffle the order of top-K candidates for each piece of data to do Data Augmentation """
        for index, candidates in enumerate(top_K_candidates):
            np.random.shuffle(candidates)
            top_K_candidates[index] = candidates

            truth = truth_label_list[index]
            # return a ground truth index indicator in candidates.
            indicator = np.isin(np.asarray(top_K_candidates[index]), np.asarray(truth)).astype(int)
            ground_truth_indicator.append(indicator)

    return top_K_candidates, ground_truth_indicator, truth_label_list, statement_list, entity_span_index_roberta

def compute_recall(pred_list, ground_truth_list):
    recall_list = []
    for index, pred in enumerate(pred_list):
        truth = ground_truth_list[index]
        overlap = len(set(pred) & set(truth))
        recall = overlap / (len(truth))
        recall_list.append(recall)
    return recall_list

def compute_precision(pred_list, ground_truth_list):
    precision_list = []
    for index, pred in enumerate(pred_list):
        truth = ground_truth_list[index]
        overlap = len(set(pred) & set(truth))
        precision = overlap / (len(pred) + 1e-6)
        precision_list.append(precision)
    return precision_list

def compute_f1(pred_list, ground_truth_list):
    recall_list = compute_recall(pred_list, ground_truth_list)
    precision_list = compute_precision(pred_list, ground_truth_list)
    f1_list = []
    for recall, precision in zip(recall_list, precision_list):
        f1 = 2 * (recall * precision) / (recall + precision + 1e-6)
        f1_list.append(f1)
    return recall_list, precision_list, f1_list

def groupby_label(label_indicator, len_test):
    pred_label_index = []

    label_for_current_data = []
    p = 0
    for i in range(len_test):
        while label_indicator[p][0] == i:
            label_for_current_data.append(label_indicator[p][1])
            p = p + 1
            if p == len(label_indicator):
                break

        pred_label_index.append(label_for_current_data)
        label_for_current_data = [] 
    
    return pred_label_index


def macro(gold, pred, len_test):
    """ adopt from UFET paper codes """
    def f1(p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float(p + r)
    num_examples = len_test
    p = 0.
    r = 0.
    pred_example_count = 0.
    pred_label_count = 0.
    gold_label_count = 0.
    for true_labels, predicted_labels in zip(gold, pred):
        if predicted_labels:
            pred_example_count += 1
            pred_label_count += len(predicted_labels)
            per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            p += per_p
        if len(true_labels):
            gold_label_count += 1
            per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
            r += per_r
    if pred_example_count > 0:
        precision = p / pred_example_count
    if gold_label_count > 0:
        recall = r / gold_label_count
    avg_elem_per_pred = pred_label_count / pred_example_count
    return num_examples, pred_example_count, avg_elem_per_pred, precision, recall, f1(precision, recall)


def metric(threshold_list, preds, top_K_candidates, truth_label_list, class_map, len_set):
    macro_performance_list = []
    for _, th in enumerate(tqdm(threshold_list, desc = 'searching threshold')):
        try:
            label_indicator = np.array(np.where(preds>th)) # for each piece of data, the indice of all the labels are predicted as truth 
            label_indicator = np.concatenate((label_indicator[0].reshape(-1, 1), label_indicator[1].reshape(-1, 1)), axis = 1)  # reshape to fit groupby function
            
            pred_label_index = groupby_label(label_indicator, len_set) # group by the label_indicator's first column. It is the predictions for each piece of data

            pred_results = []
            for top_k, label_index in zip(top_K_candidates, pred_label_index):
                pred_results.append(top_k[label_index])

            pred_results_index = [[class_map[i] for i in j] for j in pred_results]
            test_ground_truth_class_id = [[class_map[i] for i in j] for j in truth_label_list]

            recall_list, precision_list, f1_list = compute_f1(pred_results_index, test_ground_truth_class_id)

            avg_recall = sum(recall_list)/len(recall_list)
            avg_precision = sum(precision_list)/len(precision_list)
            avg_f1 = sum(f1_list)/len(f1_list)

            avg_performance = '(\navg recall: ' + str(avg_recall) + '\navg precision: ' + str(avg_precision) + '\navg f1: ' + str(avg_f1) + '\n)'
            # print(avg_performance)

            count, pred_count, avg_pred_count, macro_precision, macro_recall, macro_f1 = macro(test_ground_truth_class_id, pred_results_index, len_set)

            macro_performance = (('threshold', f"{th:.4}"), ('P', f"{macro_precision:.2%}"), ('R', f"{macro_recall:.2%}"), ('F1', f"{macro_f1:.2%}"))
            # print(macro_performance)
            
            macro_performance_list.append(macro_performance)
            
        except:
            pass
    macro_performance_list.sort(key = lambda x: -float(x[3][1].replace('%', 'e-2')))
    return macro_performance_list

def evaluate(model, dev_dataloader, test_dataloader, device, dev_top_K_candidates, test_top_K_candidates, class_map, dev_truth_label_list, test_truth_label_list, args):
    """ evaluate """
    model.eval()    
    len_test = len(test_truth_label_list)
    len_dev = len(dev_truth_label_list)
    """ get pred for dev data """
    dev_preds = []
    for _, batch in enumerate(tqdm(dev_dataloader, desc="load dev data")):
        input_ids, input_mask, segment_ids, label_ids, class_segment_ids, entity_span_index = batch
        
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, input_mask, segment_ids, entity_span_index, args.embedding_method)

        if len(dev_preds) == 0:
            dev_preds.append(logits.detach().cpu().numpy())
        else:
            dev_preds[0] = np.append(dev_preds[0], logits.detach().cpu().numpy(), axis=0)

    t = 0.5
    dev_preds = dev_preds[0].reshape(dev_preds[0].shape[0], -1)
    dev_preds = softmax(dev_preds/t,axis=1)

    """ get pred for test data """
    test_preds = []
    for _, batch in enumerate(tqdm(test_dataloader, desc="load test data")):
        input_ids, input_mask, segment_ids, label_ids, class_segment_ids, entity_span_index = batch
        
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, input_mask, segment_ids, entity_span_index, args.embedding_method)

        if len(test_preds) == 0:
            test_preds.append(logits.detach().cpu().numpy())
        else:
            test_preds[0] = np.append(test_preds[0], logits.detach().cpu().numpy(), axis=0)

   
    test_preds = test_preds[0].reshape(test_preds[0].shape[0], -1)
    test_preds = softmax(test_preds/t,axis=1)

    """ get best threshold for dev data """
    dev_threshold_list = np.arange(args.threshold[0], args.threshold[1], 0.0001)
    dev_macro_performance_list = metric(dev_threshold_list, dev_preds, dev_top_K_candidates, dev_truth_label_list, class_map, len_dev)

    # get best threshold according to the dev set then apply it on test set
    best_threshold = [float(dev_macro_performance_list[0][0][1])]
    
    test_macro_performance_list = metric(best_threshold, test_preds, test_top_K_candidates, test_truth_label_list, class_map, len_test)

    dev_performance = dev_macro_performance_list[0]
    test_performance = test_macro_performance_list[0]

    return dev_performance, test_performance

def main(args_train_batch_size, args_test_batch_size, args_num_train_epochs, args_learning_rate, args_ENABLE_WANDB, args_K, args_embedding_method, args_seed, args_eval_each_epoch, args_N, args_max_seq_length, args_threshold):

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
    parser.add_argument("--embedding_method",
                        default=args_embedding_method,
                        type=str,
                        help="Use mean or sum to get embeddings")
    parser.add_argument("--K",
                        default=args_K,
                        type=int,
                        help="Total number of top candidates selected")  
    parser.add_argument("--N",
                        default=args_N,
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
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--threshold",
                        default=args_threshold,
                        type=str,
                        help="Threshold range.")

    parser.add_argument('-f')
 
    args = parser.parse_args()   

    args.threshold = [float(i) for i in args.threshold.split(',')] 
    device = torch.device("cuda")

    """ set random seed """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    """ load top-K selection model """
    # One BERT to encode statements
    statement_model = AutoModel.from_pretrained('bert-base-uncased', local_files_only = True).to(device)
    statement_model.load_state_dict(torch.load(pretrained_statement_model_path))
    statement_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, local_files_only = True)
    statement_model.to(device)
    statement_model.eval()

    """ load data  """
    _, all_type_list, all_type_list_vanilla= db.load_data(train_path, type_path)

    train_example_list = load_all_data(train_path, type_path)
    dev_example_list = load_all_data(dev_path, type_path)
    test_example_list = load_all_data(test_path, type_path)
    class_map = {label : i for i, label in enumerate(all_type_list)}

    """ load top-k Entailment model """
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=True, local_files_only = True)

    """ Add special token <ent> to seperate entity """
    special_tokens_dict = {'additional_special_tokens': ['<ent>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

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
    train_top_K_candidates, train_ground_truth_indicator, _, train_statement_list, train_entity_span_index_roberta = get_top_K_candidates(statement_model, types_vector_path, train_example_list ,all_type_list, device, statement_tokenizer, tokenizer, args, 'train')


    dev_top_K_candidates, _, dev_truth_label_list, dev_statement_list, dev_entity_span_index_roberta = get_top_K_candidates(statement_model, types_vector_path, dev_example_list ,all_type_list, device, statement_tokenizer, tokenizer, args, 'dev')

    test_top_K_candidates, _, test_truth_label_list, test_statement_list, test_entity_span_index_roberta = get_top_K_candidates(statement_model, types_vector_path, test_example_list ,all_type_list, device, statement_tokenizer, tokenizer, args, 'test')

    test_ground_truth_class_id = [[class_map[i] for i in j] for j in test_truth_label_list]
    dev_ground_truth_class_id = [[class_map[i] for i in j] for j in dev_truth_label_list]

    print('Top-K selection')
    """ ------------------- """
    
    train_dataloader = examples_to_features('train', train_statement_list, train_entity_span_index_roberta, train_top_K_candidates, train_ground_truth_indicator, all_type_list, args, tokenizer, args.train_batch_size, dataloader_mode='random')

    dev_dataloader = examples_to_features('dev', dev_statement_list, dev_entity_span_index_roberta, dev_top_K_candidates, dev_ground_truth_class_id, all_type_list, args, tokenizer, args.test_batch_size, dataloader_mode='sequential')

    test_dataloader = examples_to_features('test', test_statement_list, test_entity_span_index_roberta, test_top_K_candidates, test_ground_truth_class_id, all_type_list, args, tokenizer, args.test_batch_size, dataloader_mode='sequential')

    """ training """
    performence_each_epoch = []
    for epoch in range(args.num_train_epochs):
        for _, batch in enumerate(tqdm(train_dataloader, desc='train|epoch'+str(epoch))):
            model.train()
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids, label_ids, class_segment_ids, entity_span_index = batch

            logits = model(input_ids, input_mask, segment_ids, entity_span_index, args.embedding_method)

            bcsz= input_ids.shape[0]

            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(bcsz, -1), label_ids.to(device).float())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if args.eval_each_epoch:
            dev_performance, test_performance = evaluate(model, dev_dataloader, test_dataloader, device, dev_top_K_candidates, test_top_K_candidates, class_map, dev_truth_label_list, test_truth_label_list, args)
            performence_each_epoch.append((dev_performance, test_performance))
            print('dev_performance:', dev_performance)
            print('test_performance:', test_performance)
            print('-------------------')

if __name__ == "__main__":
    args_train_batch_size = 8 
    args_test_batch_size = 1 #256  
    args_num_train_epochs = 0
    args_learning_rate = 1e-5
    args_ENABLE_WANDB = False
    args_K = 80
    args_N = 1
    args_embedding_method = 'mean' # 'sum' or 'mean'
    args_seed = 36
    args_result_name = ''
    args_eval_each_epoch = True
    args_max_seq_length = 450 # 320 if K is 50. 450 if K is 80
    args_threshold = '0,0.02'

    bert_hidden_dim = 1024
    pretrain_model_dir = 'roberta-large' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'

    train_path = '../data/ultrafine_acl18/release/crowd/train.json'
    dev_path = '../data/ultrafine_acl18/release/crowd/dev.json'
    test_path = '../data/ultrafine_acl18/release/crowd/test.json'
    
    type_path = '../data/ultrafine_acl18/release/ontology/types.txt'
    types_vector_path = '../data/ultrafine_acl18/types_vector_768.txt'
    pretrained_statement_model_path = './7728_model.pth'

    main(args_train_batch_size, args_test_batch_size, args_num_train_epochs, args_learning_rate, args_ENABLE_WANDB, args_K, args_embedding_method, args_seed, args_eval_each_epoch, args_N, args_max_seq_length, args_threshold)