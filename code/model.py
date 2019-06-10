import torch
from torch import nn
import torch.nn.functional as F

'''
    此文件构架模型并实现正向传播过程
'''

'''
    Embedding:
        torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, weight=None)
        input:LongTensor of arbitrary shape containing the indices to extract
        Output: (*, embedding_dim), where * is the input shape
    GRU:
        torch.nn.GRU(*args, **kwargs)
        input_size hidden_size 
        batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)
        inputs: 
                input (seq_len, batch, input_size)
                h_0  (num_layers * num_directions, batch, hidden_size)
        outputs: 
                output (seq_len, batch, num_directions * hidden_size)
                h_n (num_layers * num_directions, batch, hidden_size)
    Conv2d:
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        input: N, C, H, W
        output: N, C, H, W (N is batch_size, C is channel)
'''


class Config():
    def __init__(self, advance=0):
        self.max_num_utterance = 10
        self.max_sentence_len = 20
        self.word_embedding_size = 200
        self.GRU1_hidden_size = 200  # GRU1的hidden size
        self.GRU2_hidden_size = 50  # GRU2的hidden size
        self.total_words = 218593
        self.v_length = 50
        self.advance = bool(advance)  # whether to use a new matrix (A1_matrix) which is not used in paper


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.advance = config.advance
        self.max_num_utterance = config.max_num_utterance
        self.max_sentence_len = config.max_sentence_len
        self.word_embedding_size = config.word_embedding_size
        self.GRU1_hidden_size = config.GRU1_hidden_size
        self.GRU2_hidden_size = config.GRU2_hidden_size
        self.total_words = config.total_words
        self.v_length = config.v_length

        self.word_embedding = nn.Embedding(num_embeddings=self.total_words, embedding_dim=self.word_embedding_size)
        self.word_embedding.weight.requires_grad = True

        self.utterance_GRU = nn.GRU(input_size=self.word_embedding_size, hidden_size=self.GRU1_hidden_size,
                                    bidirectional=False, batch_first=True)
        ih_r = (param.data for name, param in self.utterance_GRU.named_parameters() if 'weight_ih' in name)
        hh_r = (param.data for name, param in self.utterance_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_r:
            nn.init.orthogonal_(k)
        for k in hh_r:
            nn.init.orthogonal_(k)

        self.response_GRU = nn.GRU(input_size=self.word_embedding_size, hidden_size=self.GRU1_hidden_size,
                                   bidirectional=False, batch_first=True)
        ih_r = (param.data for name, param in self.utterance_GRU.named_parameters() if 'weight_ih' in name)
        hh_r = (param.data for name, param in self.utterance_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_r:
            nn.init.orthogonal_(k)
        for k in hh_r:
            nn.init.orthogonal_(k)

        self.conv = nn.Conv2d(2, 8, kernel_size=(3, 3))
        conv_weight = (param.data for name, param in self.conv.named_parameters() if 'weight' in name)
        for w in conv_weight:
            nn.init.kaiming_normal_(w)

        self.pool = nn.MaxPool2d((3, 3), stride=(3, 3))

        self.linear = nn.Linear(288, self.v_length)
        linear_weight = (param.data for name, param in self.linear.named_parameters() if "weight" in name)
        for w in linear_weight:
            nn.init.xavier_uniform_(w)

        self.A1_matrix = torch.nn.Parameter(
            torch.randn(self.word_embedding_size, self.word_embedding_size))
        nn.init.xavier_uniform_(self.A1_matrix)

        self.A2_matrix = torch.nn.Parameter(
            torch.randn(self.GRU1_hidden_size, self.GRU1_hidden_size))
        nn.init.xavier_uniform_(self.A2_matrix)

        self.final_GRU = nn.GRU(input_size=self.v_length, hidden_size=self.GRU2_hidden_size, bidirectional=False,
                                batch_first=True)
        ih_f = (param.data for name, param in self.final_GRU.named_parameters() if 'weight_ih' in name)
        hh_f = (param.data for name, param in self.final_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_f:
            nn.init.orthogonal_(k)
        for k in hh_f:
            nn.init.orthogonal_(k)

        self.W_1_1 = torch.nn.Parameter(
            torch.randn(self.GRU2_hidden_size, self.GRU1_hidden_size))
        nn.init.xavier_normal_(self.W_1_1)

        self.W_1_2 = torch.nn.Parameter(
            torch.randn(self.GRU2_hidden_size, self.GRU2_hidden_size))
        nn.init.xavier_normal_(self.W_1_2)

        self.b_1 = torch.nn.Parameter(torch.randn(self.GRU2_hidden_size))
        nn.init.normal_(self.b_1)

        self.t_s = torch.nn.Parameter(torch.randn(self.GRU2_hidden_size))
        nn.init.normal_(self.t_s)

        self.final_linear = nn.Linear(self.GRU2_hidden_size, 2)
        final_linear_weight = (param.data for name, param in self.final_linear.named_parameters() if "weight" in name)
        for w in final_linear_weight:
            nn.init.xavier_uniform_(w)

    def forward(self, utterance, response):
        '''
           utterance:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
           response:(self.batch_size, self.max_sentence_len)
        '''

        all_utterance_embeddings = self.word_embedding(utterance)
        # batch_size,max_num_utterance,max_sentence_len ->
        # batch_size,max_num_utterance,max_sentence_len,word_embedding_size

        response_embeddings = self.word_embedding(response)
        # batch_size,max_sentence_len -> batch_size,max_sentence_len,word_embedding_size

        all_utterance_embeddings = all_utterance_embeddings.permute(1, 0, 2, 3)
        # batch_size,max_num_utterance,max_sentence_len ->
        # max_num_utterance,batch_size,max_sentence_len,word_embedding_size

        response_GRU_embeddings, _ = self.response_GRU(response_embeddings)
        # batch_size,max_sentence_len,word_embedding_size ->
        # batch_size,max_sentence_len,GRU1_hidden_size

        matching_vectors = []
        hidden_states = []
        for i, utterance_embeddings in enumerate(all_utterance_embeddings):
            # utterance_embeddings (batch_size,max_sentence_len,word_embedding_size)
            # response_embeddings (batch_size,max_sentence_len,word_embedding_size)

            if self.advance:
                matrix1 = torch.einsum('abm,mn,acn->abc', utterance_embeddings, self.A1_matrix, response_embeddings)
                # matrix1 (batch_size,max_sentence_len,max_sentence_len)
            else:
                matrix1 = torch.einsum('abe,ace->abc', utterance_embeddings, response_embeddings)
                # matrix1 (batch_size,max_sentence_len,max_sentence_len)

            utterance_GRU_embeddings, hidden_state = self.utterance_GRU(utterance_embeddings)
            hidden_state = torch.squeeze(hidden_state)
            hidden_states.append(hidden_state)
            # utterance_GRU_embeddings (batch_size,max_sentence_len,word_embedding_size) ->
            # (batch_size,max_sentence_len,GRU1_hidden_size)
            # hidden_state (batch_size,hidden_size)
            # hidden_states (batch_size,hidden_size)
            if self.advance:
                matrix2 = torch.einsum('abm,mn,acn->abc', utterance_GRU_embeddings, self.A1_matrix,
                                       response_GRU_embeddings)
                # matrix2 (batch_size,max_sentence_len,max_sentence_len)
            else:
                matrix2 = torch.einsum('abe,ace->abc', utterance_GRU_embeddings, response_GRU_embeddings)
                # matrix2 (batch_size,max_sentence_len,max_sentence_len)

            matrix = torch.stack([matrix1, matrix2], dim=1)
            # matrix (batch_size,2,max_sentence_len,max_sentence_len)

            x_conv = F.relu(self.conv(matrix))
            x_pooling = self.pool(x_conv)

            pre_matching_vector = x_pooling.view(x_pooling.size(0), -1)
            matching_vector = torch.tanh(self.linear(pre_matching_vector))
            matching_vectors.append(matching_vector)

        matching_vectors_GRU, _ = self.final_GRU(torch.stack(matching_vectors, dim=1))
        # (N,max_num_utterance,v_length)->(N,max_num_utterance,GRU2_hidden_size)

        hidden_states = torch.stack(hidden_states, dim=1)  # (batch, 10,GRU1_hidden_size)
        matching_vectors_GRU = matching_vectors_GRU.permute(1, 0, 2)  # (10,batch,GRU2_hidden_size)
        hidden_states = hidden_states.permute(1, 0, 2)  # (10,batch,GRU1_hidden_size)

        t = torch.tanh(torch.einsum('ij,akj->aki', self.W_1_1, hidden_states) +
                       torch.einsum('ij,akj->aki', self.W_1_2, matching_vectors_GRU) +
                       self.b_1)
        # (10,batch_size,GRU2_hidden_size)

        pre_alpha = torch.einsum('ijk,k->ij', t, self.t_s)
        alpha = F.softmax(pre_alpha, dim=0)
        # 10,batch_size -> 10,batch_size
        L = torch.einsum('ij,ijk->jk', alpha, matching_vectors_GRU)
        # batch_size, GRU2_hidden_size
        logits = self.final_linear(L)
        y_pred = F.log_softmax(logits, dim=1)
        y_pred_pro = F.softmax(logits, dim=1)
        return y_pred, y_pred_pro
