import pickle


def txt2list4test(src_path, save_path, dic):
    count = 0
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
                    if word not in dic.keys():
                        temp_sentence[i] = dic['UNK']
                        count += 1
                    else:
                        temp_sentence[i] = dic[word]
                temp_utterance.append(temp_sentence)
    print('prepare to save list')
    with open(save_path, 'wb') as f:
        pickle.dump(dialogues, f)
    print('complete to save list')
    print('{} words can not found in dict'.format(count))


def get_dict(src_path):
    with open(src_path, 'rb') as f:
        dic = pickle.load(f, encoding='bytes')
    return dic


if __name__ == '__main__':
    dic = get_dict('../Data/word2index.pkl')
    txt2list4test('../Data/test/test_context.txt', '../Data/test/utterance.pkl', dic)
    txt2list4test('../Data/test/test_reply.txt', '../Data/test/responses.pkl', dic)
    # 14844 words can not found in dict
