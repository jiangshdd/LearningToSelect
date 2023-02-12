
import numpy as np
import torch
import json
import wandb
import argparse

from sklearn.metrics import pairwise

from torch.nn import CosineEmbeddingLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from tqdm import tqdm
from transformers import  AutoTokenizer, AutoModel
from transformers.optimization import AdamW


def load_data(data_path, type_path):

    dataset = []
    with open(data_path) as f:
        for line in f:
            dataset.append(json.loads(line))

    all_type_list = []

    with open(type_path) as f:
        for line in f:
            all_type_list.append(line.strip())

    all_type_list_vanilla = all_type_list
    all_type_list = [' '.join(i.split('_')) for i in all_type_list]
    # print(len(all_type_list))
    return dataset, all_type_list, all_type_list_vanilla

def load_test_data(data_path):

    dataset = []
    with open(data_path) as f:
        for line in f:
            dataset.append(json.loads(line))

    return dataset

def parse_data(dataset, all_type_list):
    truth_label_list = []
    truth_label_list_vanilla = []
    truth_label_index_list = []
    data_list = []  #[[statement1, statement_index], ...]
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
        
        data_list.append([statement, data_index])
        truth_label_list.append(entity_typing)
        truth_label_list_vanilla.append(entitiy_typing_vanilla)
        truth_label_index_list.append(entity_typing_index)
        
    data_list = np.array(data_list)
    truth_label_index_list = np.array(truth_label_index_list)

    return data_list, truth_label_list, truth_label_list_vanilla, truth_label_index_list

'''
[
[statement1, statement_index], 
[statement2, statemend_index]
]
'''
def tokenize_all_types(all_type_list, tokenizer):
    """ Tokenize all types """
    all_type_token = tokenizer(all_type_list, return_tensors="pt", padding = 'longest')
    return all_type_token

def tokenize_data(data, tokenizer):
    """ 
    Tokenize the statement, get the entity span index after tokenization
    so the embedding of entity can be extracted later 
    """
    statement_index = torch.tensor(data[:,1].astype(int))
    statement_token = tokenizer(list(data[:,0]), return_tensors="pt", padding = 'longest')

    # use {} to seperate entity span
    entity_sep_left = tokenizer.encode('{')[1]
    entity_sep_right = tokenizer.encode('}')[1]
    
    input_ids = statement_token.input_ids

    # locate '{' and '}' to find the location of entity span
    entity_sep_left_index = torch.where(input_ids== entity_sep_left)[1]
    entity_sep_right_index = torch.where(input_ids == entity_sep_right)[1] -1
    span_start_index = entity_sep_left_index
    span_end_index = entity_sep_right_index 
    entity_span_index = torch.transpose(torch.stack((span_start_index, span_end_index)), 1, 0)

    def remove_entity_sep(inputs, entity_sep_left_index, entity_sep_right_index):
        """ remove '{' and '}' """
        mask = torch.ones_like(inputs).scatter_(1, entity_sep_left_index.unsqueeze(1), 0.)
        inputs = inputs[mask.bool()].view(inputs.shape[0], inputs.shape[1]-1)
    
        mask = torch.ones_like(inputs).scatter_(1, entity_sep_right_index.unsqueeze(1), 0.)
        inputs = inputs[mask.bool()].view(inputs.shape[0], inputs.shape[1]-1)
        return inputs

    statement_token.input_ids = remove_entity_sep(statement_token.input_ids, entity_sep_left_index, entity_sep_right_index)
    statement_token.attention_mask = remove_entity_sep(statement_token.attention_mask, entity_sep_left_index, entity_sep_right_index)

    # for idx, i in enumerate(entity_span_index):
    #     entity_decode = tokenizer.decode(statement_token.input_ids[idx][i[0]:i[1]])
    #     print(entity_decode)
    return statement_token, entity_span_index, statement_index

def examples_to_features(data, tokenizer, batch_size, dataloader_mode):

    source_features, all_entity_span_index, all_statement_index  = tokenize_data(data, tokenizer)

    all_input_ids = torch.stack([f for f in source_features.input_ids])
    all_input_mask = torch.stack([f for f in source_features.attention_mask])

    data_tensor = TensorDataset(all_input_ids, all_input_mask, all_entity_span_index, all_statement_index)
    
    if dataloader_mode=='sequential':
        sampler = SequentialSampler(data_tensor)
    else:
        sampler = RandomSampler(data_tensor)
    dataloader = DataLoader(data_tensor, sampler=sampler, batch_size=batch_size)


    return dataloader

def get_entity_embedding(last_hidden_state, entity_span_index, flag = 'mean'):
    entity_embedding_list = []
    for hidden, span in zip(last_hidden_state, entity_span_index):
        if flag == 'mean':
            entity_embedding =  torch.mean(hidden[span[0]:span[1]], 0)
            entity_embedding =  torch.mean(torch.stack((entity_embedding, hidden[0])), 0)
        elif flag == 'sum':
            entity_embedding =  torch.sum(hidden[span[0]:span[1]], 0)
            entity_embedding =  torch.mean(torch.stack((entity_embedding, hidden[0])), 0)
        entity_embedding_list.append(entity_embedding)
    return torch.stack(entity_embedding_list)


def get_label_embedding(last_hidden_state, label_attention_mask, flag = 'mean'):
    label_embedding_list = []
    for hidden, mask in zip(last_hidden_state, label_attention_mask):
        hidden = hidden[mask.bool()]
        if flag == 'mean':
            label_embedding = torch.mean(hidden[1:-1], 0)
        elif flag == 'sum':
            label_embedding = torch.sum(hidden[1:-1], 0)
        label_embedding_list.append(label_embedding)
    return torch.stack(label_embedding_list)

def get_optimizer(model, learning_rate):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    return AdamW(optimizer_grouped_parameters, lr=learning_rate)

def load_type_vector(types_vector_path):
    # load all type vectors
    type_embedding_dict = {}
    all_types_list_loaded = []
    all_types_embedding_loaded = []
    with open(types_vector_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            type = values[0]
            vector = np.asarray(values[1:], "float32")
            type_embedding_dict[type] = vector
            all_types_list_loaded.append(type)
            all_types_embedding_loaded.append(vector)
    all_types_embedding_loaded = torch.tensor(all_types_embedding_loaded)
    return type_embedding_dict, all_types_list_loaded, all_types_embedding_loaded


def embedding_similarity(entity_embeddings, type_embeddings):
    entity_embeddings = entity_embeddings.detach()
    similarity = pairwise.cosine_similarity(entity_embeddings.cpu(), type_embeddings.cpu())
    return similarity

def compute_recall(top_k_list, ground_truth_list):
    recall_list = []
    for index, top_k in enumerate(top_k_list):
        truth = ground_truth_list[index]
        overlap = len(set(top_k) & set(truth))
        recall = overlap / (len(truth))
        recall_list.append(recall)
    return recall_list

def sample_negative_type(negative_label_in_batch, all_type_token):
    
    neg_label_input_ids = torch.cat([all_type_token.input_ids[negative_label_in_batch[i]] for i in range(negative_label_in_batch.shape[0])])
    neg_label_input_mask = torch.cat([all_type_token.attention_mask[negative_label_in_batch[i]] for i in range(negative_label_in_batch.shape[0])])

    return neg_label_input_ids, neg_label_input_mask

def sample_positive_type(truth_label_in_batch, all_type_token):

    pos_label_input_ids = torch.cat([all_type_token.input_ids[truth_label_in_batch[i]] for i in range(truth_label_in_batch.shape[0])])
    pos_label_input_mask = torch.cat([all_type_token.attention_mask[truth_label_in_batch[i]] for i in range(truth_label_in_batch.shape[0])])

    return pos_label_input_ids, pos_label_input_mask

def align_pairs(label_in_batch, entity_embedding_list, device, flag = 'positive'):
    """ 
    Repeat each statement in the batch to match its corresponding labels to construct pos/neg pairs. Num of pos/neg pairs for each statment == Num of pos/neg labels
    """
    num_pair = torch.tensor([len(i) for i in label_in_batch]).to(device)

    entity_embedding = torch.repeat_interleave(entity_embedding_list, num_pair, dim=0)
    target = torch.ones(entity_embedding.shape[0], dtype=int).to(device)
    if flag == 'negative':
        target = torch.neg(target)

    return entity_embedding, target

def get_top_negative_index(label_model, negative_label_in_batch, all_type_token, entity_embedding_list, N_neg_sample, embedding_method):
    """ 
    For each entity, sample its top similar negative labels to construct the negative pairs 

    N_neg_sample: control the number of samples
    """
    # Get embeddings for all types
    with torch.no_grad():
        all_type_outputs = label_model(all_type_token.input_ids, all_type_token.attention_mask)
    all_type_embeddings = get_label_embedding(all_type_outputs[0], all_type_token.attention_mask, embedding_method)
   
    # For each entity, compute its similarity between all types, then extract similarities of negative types
    similarity = embedding_similarity(entity_embedding_list, all_type_embeddings)
    neg_similarity = [similarity[i][negative_label_in_batch[i]] for i in range(negative_label_in_batch.shape[0])]

    # Sample top #N_neg_sample similar negative labels
    top_indice = np.array([np.argsort(i)[-N_neg_sample:] for i in neg_similarity])

    return top_indice

def save_model(statement_model, label_model, statement_model_save_dir, label_model_save_dir, args):
    """ Saving model """
    print('Saving models ...')
    torch.save(statement_model.state_dict(), statement_model_save_dir + args.statement_model_save_path)
    torch.save(label_model.state_dict(), label_model_save_dir + args.label_model_save_path)

def write_type_vector(statement_model, label_model, all_type_token, types_vector_path, all_type_list_vanilla, args):
    """ Writing label vector into a file """
    statement_model.eval()
    label_model.eval()

    all_type_input_ids = all_type_token.input_ids
    all_type_attention_mask = all_type_token.attention_mask
    with torch.no_grad():
        all_type_outputs = label_model(all_type_input_ids, all_type_attention_mask)
    all_type_embeddings = get_label_embedding(all_type_outputs[0], all_type_attention_mask, args.embedding_method)
    with open(types_vector_path, 'w') as f:
        for type, vector in tqdm(zip(all_type_list_vanilla, all_type_embeddings), total=len(all_type_embeddings), desc='writing vectors'):
            f.write(str(type) + ' ')
            f.write(' '.join([str(i) for i in vector.tolist()]) + '\n')    

def evaluate(statement_model, types_vector_path, all_type_list, device, test_path, tokenizer, args, result_dir, flag):
    # load all types embeddings
    type_embedding_dict, all_types_list_loaded, all_types_embedding_loaded = load_type_vector(types_vector_path)
    test_data_list, _ , test_truth_label_list, _ = parse_data(load_test_data(test_path), all_type_list)
    test_statement_dataloader = examples_to_features(test_data_list, tokenizer, len(test_data_list), 'sequential')
    
    # Get test entity embeddings
    for _, batch in enumerate(tqdm(test_statement_dataloader, desc='Evaluate')):
        batch = tuple(t.to(device) for t in batch)
        # Get input_ids(tokens) for statements
        statement_input_ids, statement_input_mask, entity_span_index, statement_index = batch

        # Get embeddings for entity span in each statement
        with torch.no_grad():
            statement_outputs = statement_model(statement_input_ids, statement_input_mask)
        test_entity_embedding_list = get_entity_embedding(statement_outputs[0], entity_span_index, args.embedding_method)
    
    # For each entity, compute its similarity between all types, then select top K types
    similarity = embedding_similarity(test_entity_embedding_list, all_types_embedding_loaded)
    all_recall = []
    for top_k in args.K:
        top_K_indice = [np.argsort(i)[-top_k:] for i in similarity]
        top_K_candidates = [np.array(all_types_list_loaded)[i] for i in top_K_indice]

        # compute recall
        recall_list = compute_recall(top_K_candidates, test_truth_label_list)
        # for index, i in enumerate(recall_list):
        #     print(i, len(top_K_candidates[index]))
        avg_recall = sum(recall_list)/len(recall_list)
        all_recall.append(('Top ' + str(top_k), avg_recall))
    
    # print('all_recall', all_recall)   

    return all_recall

def main(args_train_batch_size, args_num_train_epochs, args_learning_rate, args_ENABLE_WANDB, args_K, args_N_neg_sample, args_embedding_method, args_seed, args_statement_model_save_path, args_label_model_save_path, args_sample_top_neg, args_eval_each_epoch):

    parser = argparse.ArgumentParser()


    parser.add_argument("--statement_model_save_path",
                        default=args_statement_model_save_path,
                        type=str)  
    parser.add_argument("--label_model_save_path",
                        default=args_label_model_save_path,
                        type=str)  
    parser.add_argument("--train_batch_size",
                        default=args_train_batch_size,
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
    parser.add_argument("--N_neg_sample",
                        default=args_N_neg_sample,
                        type=int,
                        help="Total number of negative pair sampled for each entity.")
    parser.add_argument("--sample_top_neg",
                        default=args_sample_top_neg,
                        action='store_true',
                        help="For each entity, sample its top similar negative labels to construct the negative pairs. If set to False, do random sample")
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
                        type=str,
                        help="Total number of top candidates selected")  
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

    parser.add_argument('-f')
 
    args = parser.parse_args()   

    args.K = [int(i) for i in args.K.split(',')] 
    label_model_save_path_vanilla = args.label_model_save_path
    statement_model_save_path_vanilla = args.statement_model_save_path

    train_path = 'G:/My Drive/UIC/Research/top-K-entailment/data/ultrafine_acl18/release/crowd/train.json'
    dev_path = 'G:/My Drive/UIC/Research/top-K-entailment/data/ultrafine_acl18/release/crowd/dev.json'
    test_path = 'G:/My Drive/UIC/Research/top-K-entailment/data/ultrafine_acl18/release/crowd/test.json'

    type_path = 'G:/My Drive/UIC/Research/top-K-entailment/data/ultrafine_acl18/release/ontology/types.txt'
    types_vector_path = 'G:/My Drive/UIC/Research/top-K-entailment/data/types_vector_768-test.txt'
    statement_model_save_dir = 'G:/My Drive/UIC/Research/top-K-entailment/saved_model/statement_model/'
    label_model_save_dir = 'G:/My Drive/UIC/Research/top-K-entailment/saved_model/label_model/'
    result_dir = 'G:/My Drive/UIC/Research/top-K-entailment/results_top_k/'
    pretrain_model_dir = 'bert-base-cased'
    # pretrain_model_dir = 'roberta-large'

    device = torch.device("cuda")

    if args.ENABLE_WANDB:
        wandb.setup(wandb.Settings(program="dual_bert.py", program_relpath="dual_bert.py"))
        wandb.init(project="dual-bert", entity="jiangshd")

    # Two different BERT models
    # One BERT to encode statements
    statement_model = AutoModel.from_pretrained(pretrain_model_dir, local_files_only = True).to(device)

    # # train from last check point
    # statement_model.load_state_dict(torch.load(statement_model_save_dir + 'ep2-model-2022-04-03T11.32.37.3915935-05.00.pth'))
    # Another BERT to encode labels
    label_model = AutoModel.from_pretrained(pretrain_model_dir, local_files_only = True).to(device)
    # # train from last check point
    # label_model.load_state_dict(torch.load(label_model_save_dir + 'ep2-model-2022-04-03T11.32.37.3915935-05.00.pth'))

    statement_optimizer = get_optimizer(statement_model, args.learning_rate)
    label_optimizer = get_optimizer(label_model, args.learning_rate)

    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=True, local_files_only = True)

    dataset, all_type_list, all_type_list_vanilla= load_data(train_path, type_path)
    data_list, truth_label_list, _, truth_label_index_list = parse_data(dataset, all_type_list)

    all_type_token = tokenize_all_types(all_type_list, tokenizer).to(device)
    statement_dataloader = examples_to_features(data_list, tokenizer, args.train_batch_size, 'random')

    """ Training """
    recall_each_epoch = []
    for epoch in range(args.num_train_epochs):
        for _, batch in enumerate(tqdm(statement_dataloader, desc='train|epoch_'+str(epoch))):
            statement_model.train()
            label_model.train()

            batch = tuple(t.to(device) for t in batch)

            # Get input_ids(tokens) for statements
            statement_input_ids, statement_input_mask, entity_span_index, statement_index = batch

            # Get embeddings for entity span in each statement
            statement_outputs = statement_model(statement_input_ids, statement_input_mask)
            entity_embedding_list = get_entity_embedding(statement_outputs[0], entity_span_index, args.embedding_method)

            # Get pos/neg label index of each statement in the current batch
            truth_label_in_batch = truth_label_index_list[statement_index.to('cpu').numpy()]
            all_type_index = np.arange(len(all_type_list))
            negative_label_in_batch = np.array([np.setdiff1d(all_type_index, np.array(i)) for i in truth_label_in_batch])

            if args.sample_top_neg:
                # For each entity, sample its top similar negative labels to construct the negative pairs
                negative_label_in_batch = get_top_negative_index(label_model, negative_label_in_batch, all_type_token, entity_embedding_list, args.N_neg_sample, args.embedding_method) 
            else:  
                # random sample negative pairs
                negative_label_in_batch = np.array([np.random.choice(i, args.N_neg_sample) for i in negative_label_in_batch])

            # Get input_ids(tokens) for pos labels
            pos_label_input_ids, pos_label_input_mask = sample_positive_type(truth_label_in_batch, all_type_token)

            # Get embeddings for pos labels
            pos_label_outputs = label_model(pos_label_input_ids, pos_label_input_mask)
            pos_label_embedding = get_label_embedding(pos_label_outputs[0], pos_label_input_mask, args.embedding_method)

            # align pos pairs
            pos_entity_embedding, pos_target = align_pairs(truth_label_in_batch, entity_embedding_list, device, 'positive')
            
            # Get input_ids(tokens) for negative labels
            neg_label_input_ids, neg_label_input_mask = sample_negative_type(negative_label_in_batch, all_type_token)

            # Get embeddings for neg labels
            neg_label_outputs = label_model(neg_label_input_ids, neg_label_input_mask)
            neg_label_embedding = get_label_embedding(neg_label_outputs[0], neg_label_input_mask, args.embedding_method)

            # align neg pairs
            neg_entity_embedding, neg_target = align_pairs(negative_label_in_batch, entity_embedding_list, device, 'negative')

            # concate pos and neg pairs
            all_entity_embedding = torch.cat((pos_entity_embedding, neg_entity_embedding), dim = 0)
            all_label_embedding = torch.cat((pos_label_embedding, neg_label_embedding), dim = 0)
            all_target = torch.cat((pos_target, neg_target), dim = 0)

            loss_fct = CosineEmbeddingLoss(reduction='sum')
            loss = loss_fct(all_entity_embedding, all_label_embedding, all_target)

            if args.ENABLE_WANDB == True:
                wandb.log({"loss": loss})

            loss.backward()

            statement_optimizer.step()
            statement_optimizer.zero_grad()

            label_optimizer.step()
            label_optimizer.zero_grad()

        if args.eval_each_epoch:
            """ Saving model at this epoch """
            args.statement_model_save_path = str('ep' + str(epoch) + '-') + statement_model_save_path_vanilla
            args.label_model_save_path = str('ep' + str(epoch) + '-') + label_model_save_path_vanilla
            save_model(statement_model, label_model, statement_model_save_dir, label_model_save_dir, args)
            write_type_vector(statement_model, label_model, all_type_token, types_vector_path, all_type_list_vanilla, args)
            dev_recall = evaluate(statement_model, types_vector_path, all_type_list, device, dev_path, tokenizer, args, result_dir, 'dev')
            test_recall = evaluate(statement_model, types_vector_path, all_type_list, device, test_path, tokenizer, args, result_dir, 'test')
            print('dev_recall:', dev_recall)
            print('test_recall:', test_recall)
            
            recall_each_epoch.append((dev_recall, test_recall))


if __name__ == "__main__":
    args_train_batch_size = 16  # max: 128 when N_neg_sample is 1
    args_num_train_epochs = 5
    args_learning_rate = 1e-5
    args_ENABLE_WANDB = True
    args_K = '100, 50, 20'
    args_N_neg_sample = 200
    args_sample_top_neg = False
    args_embedding_method = 'mean' # 'sum' or 'mean'
    args_seed = 32
    args_statement_model_save_path = 'model.pth'
    args_label_model_save_path = 'model.pth'
    args_result_name = ''
    args_eval_each_epoch = True

    main(args_train_batch_size, args_num_train_epochs, args_learning_rate, args_ENABLE_WANDB, args_K, args_N_neg_sample, args_embedding_method, args_seed, args_statement_model_save_path, args_label_model_save_path, args_sample_top_neg, args_eval_each_epoch)