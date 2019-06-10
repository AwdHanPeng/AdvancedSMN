'''
    此文件用于探索样本规模
'''


def explore(file):
    with open(file, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        print(lines.count('\n'))
        sum_num_utterance = 0
        sum_sentence_len = 0
        sum_session = 0
        sum_sentence = 0
        temputter = []
        tempsente = []
        for line in lines:
            if line == '\n':
                sum_session += 1
                sum_num_utterance += len(temputter)
                temputter = []
            else:
                sum_sentence += 1
                tempsente = line.replace('\n', '').split(' ')
                sum_sentence_len += len(tempsente)
                temputter.append(tempsente)
        print('平均每个session几个句子{}，平均每句话几个单词{}'.format(sum_num_utterance / sum_session, sum_sentence_len / sum_sentence))


#
# explore('../Data/test/test_context.txt')  # 10000 平均每个session几个句子3.1041，平均每句话几个单词7.397989755484682
# explore('../Data/train/train_session.txt')  # 3924369 平均每个session几个句子4.096359440205546，平均每句话几个单词6.934299582299574
# explore('../Data/valid/valid_context.txt')  # 10000 28 103 平均每个session几个句子3.1002，平均每句话几个单词7.296367976259596
explore('../Data/valid/valid_reply.txt')  # 10000 10 102 平均每个session几个句子10.0，平均每句话几个单词6.0973
