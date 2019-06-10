import pickle
import random

PAD_token = 0
UNK_token = 1
'''
    此文件构建了词袋 并依据词袋将所有输入样本进行了转换
'''


class Voc:
    def __init__(self):
        self.word2index = {'PAD': PAD_token, 'UNK': UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", UNK_token: 'UNK'}
        self.num_words = 2

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def get_index(self, word):
        return self.word2index[word]

    def save_word2index(self):
        with open('../Data/word2index.pkl', 'wb') as f:
            pickle.dump(self.word2index, f)
        print('complete to save word2index len = {}'.format(self.num_words))
        with open('../Data/index2word.pkl', 'wb') as f:
            pickle.dump(self.index2word, f)
        print('complete to save index2word len = {}'.format(self.num_words))


def ground2index():
    with open('../Data/valid/valid_ground.txt', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        index = []
        for line in lines:
            num = line.replace('\n', '')
            num = int(num)
            index.append(num)
    with open('../Data/valid/index.pkl', 'wb') as f:
        pickle.dump(index, f)
    print('complete to create and save index.pkl from valid_ground.txt')


def txt2list4valid(src_path, save_path):
    dialogues = []
    temp_utterance = []
    count = 0
    with open(src_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                dialogues.append(temp_utterance)
                count += 1
                if count % 1000 == 0:
                    print('{}*1000 sessions complete'.format(count / 1000))
                temp_utterance = []
            else:
                temp_sentence = line.replace('\n', '').split(' ')
                for i, word in enumerate(temp_sentence):
                    voc.add_word(word)
                    temp_sentence[i] = voc.get_index(word)
                temp_utterance.append(temp_sentence)
    print('prepare to save list')
    with open(save_path, 'wb') as f:
        pickle.dump(dialogues, f)
    print('complete to save list')


def txt2list4train(src_path, save_path):
    '''
    :param src_path: train_session.txt
    :param save_path: utterance.pkl correct_response.pkl
    :return:
    '''
    dialogues = []
    temp_utterance = []
    count = 0
    responses = []
    with open(src_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            if line == '\n':
                temp_sentence = temp_utterance.pop()
                responses.append(temp_sentence)
                dialogues.append(temp_utterance)
                count += 1
                if count % 1000 == 0:
                    print('{}*1000 sessions complete'.format(count / 1000))
                temp_utterance = []
                if count >= num_sessions:
                    break
            else:
                temp_sentence = line.replace('\n', '').split(' ')
                for i, word in enumerate(temp_sentence):
                    voc.add_word(word)
                    temp_sentence[i] = voc.get_index(word)
                temp_utterance.append(temp_sentence)
    print('prepare to save list')
    utterance_path, correct_response_path, error_response_path = save_path
    with open(utterance_path, 'wb') as f:
        pickle.dump(dialogues, f)
    print('complete to save utterance')
    with open(correct_response_path, 'wb') as f:
        pickle.dump(responses, f)
    print('complete to save correct_responses')
    random.shuffle(responses)
    with open(error_response_path, 'wb') as f:
        pickle.dump(responses, f)
    print('complete to save error_responses')


if __name__ == '__main__':
    num_sessions = 500000
    voc = Voc()
    txt2list4train('../Data/train/train_session.txt',
                   ['../Data/train/utterance.pkl', '../Data/train/correct_response.pkl',
                    '../Data/train/error_response.pkl'])
    txt2list4valid('../Data/valid/valid_context.txt', '../Data/valid/utterance.pkl')
    txt2list4valid('../Data/valid/valid_reply.txt', '../Data/valid/responses.pkl')
    voc.save_word2index()
    ground2index()
