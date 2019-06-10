import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

best_advance_model = 'model6'
best_primitive_model = 'Primitive5'


def padding_utterance(utterance, max_num_utterance=10):
    while len(utterance) < max_num_utterance:
        utterance.insert(0, [0])
    while len(utterance) > max_num_utterance:
        utterance.pop(0)


def padding_sentence(sentence, max_sentence_len=20):
    while len(sentence) < max_sentence_len:
        sentence.append(0)
    while len(sentence) > max_sentence_len:
        sentence.pop()


def mean_reciprocal_rank(ranks):
    sum = 0
    for rank in ranks:
        sum += (1 / rank)
    return sum / len(ranks)


def rank(lis, idx):
    '''
    查看在lis中 idx位置的数 从大到小排列为第几个数字
    +1 防止等于0
    '''
    temp_list = list(lis)  # 保证不会对形参产生影响
    temp = temp_list[idx]
    temp_list.sort(reverse=True)
    return temp_list.index(temp) + 1


def count_for_r(lis, idx, num):
    '''
    在lis中寻找前num大的值的索引，判断idx是否在这些索引之中 是则返回1，否则返回0
    '''
    temp_list = list(lis)  # 保证不会对形参产生影响
    temp = []
    for i in range(num):
        x = temp_list.index(max(temp_list))
        temp.append(x)
        temp_list[x] = 0

    if idx in temp:
        return 1
    else:
        return 0


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (utterance, response, label) in enumerate(train_loader):
        utterance, response, label = utterance.to(device), response.to(device), label.to(device)
        optimizer.zero_grad()
        output, _ = model(utterance, response)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * len(utterance), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def valid(args, model, device, valid_loader):
    # MRR MAP P@1 R10@1 R10@2 R10@5
    model.eval()
    correct_1 = 0
    correct_2 = 0
    correct_5 = 0
    with torch.no_grad():
        ranks = []
        for batch_idx, (utterance, responses, correct_indexes) in enumerate(valid_loader):
            responses = responses.permute(1, 0, 2)  # ->10,batch_size,self.max_sentence_len
            probabilities = []
            correct_indexes = list(correct_indexes)
            for response in responses:
                utterance, response = utterance.to(device), response.to(device)
                # utterance: batch_size,max_num_utterance,self.max_sentence_len
                # response: batch_size,self.max_sentence_len
                _, output = model(utterance, response)  # output :(batch_size,2)
                output = output.permute(1, 0)  # output :(2, batch_size)
                positive_probability = output[1, :]
                positive_probability = torch.squeeze(positive_probability)  # positive_probability (batch_size)
                probabilities.append(list(positive_probability))
            # probabilities 10*batch_size (list)
            probabilities = list(torch.tensor(probabilities).permute(1, 0))
            # probabilities batch_size*10 (list)

            for i in range(args.valid_batch_size):
                pro_per_sample = probabilities[i]
                correct_index = correct_indexes[i]
                # pro_per_sample 10 list
                # correct_index scalar
                correct_index = correct_index.item()
                pro_per_sample = list(pro_per_sample)
                r = rank(pro_per_sample, correct_index)
                correct_2 += count_for_r(pro_per_sample, correct_index, 2)
                correct_5 += count_for_r(pro_per_sample, correct_index, 5)
                ranks.append(r)
                if pro_per_sample.index(max(pro_per_sample)) == correct_index:
                    correct_1 += 1

            print('Valid at sample {} [{}/{} ({:.0f}%)]'.format(batch_idx * args.valid_batch_size,
                                                                batch_idx * args.valid_batch_size,
                                                                len(valid_loader.dataset),
                                                                100. * batch_idx * args.valid_batch_size / len(
                                                                    valid_loader.dataset)))
        MRR = mean_reciprocal_rank(ranks)
        MAP = MRR
        P_1 = correct_1 / len(valid_loader.dataset)
        R10_1 = P_1
        R10_2 = correct_2 / len(valid_loader.dataset)
        R10_5 = correct_5 / len(valid_loader.dataset)

    print(
        '\nvalid set: MRR = MAP = {:.4f}.P_1 = R10_1 = {:.4f}.R10_2 = {:.4f}.R10_5 = {:.4f}\n'.format(MAP, R10_1, R10_2,
                                                                                                      R10_5))
    return MAP, R10_1, R10_2, R10_5


def test(args, model, device, test_loader):
    model.eval()
    best_index = []
    ranks = []
    with torch.no_grad():
        ranks = []
        for batch_idx, (utterance, responses) in enumerate(test_loader):
            responses = responses.permute(1, 0, 2)  # ->10,batch_size,self.max_sentence_len
            probabilities = []
            for response in responses:
                utterance, response = utterance.to(device), response.to(device)
                # utterance: batch_size,max_num_utterance,self.max_sentence_len
                # response: batch_size,self.max_sentence_len
                _, output = model(utterance, response)  # output :(batch_size,2)
                output = output.permute(1, 0)  # output :(2, batch_size)
                positive_probability = output[1, :]
                positive_probability = torch.squeeze(positive_probability)  # positive_probability (batch_size)
                probabilities.append(list(positive_probability))
            # probabilities 10*batch_size (list)
            probabilities = list(torch.tensor(probabilities).permute(1, 0))
            # probabilities batch_size*10 (list)

            for i in range(args.test_batch_size):
                pro_per_sample = probabilities[i]
                # pro_per_sample 10 list
                pro_per_sample = list(pro_per_sample)
                best_index.append(pro_per_sample.index(max(pro_per_sample)))
                ranks.append(sort(pro_per_sample))

            print('Test at sample {} [{}/{} ({:.0f}%)]'.format(batch_idx * args.test_batch_size,
                                                               batch_idx * args.test_batch_size,
                                                               len(test_loader.dataset),
                                                               100. * batch_idx * args.test_batch_size / len(
                                                                   test_loader.dataset)))
    return best_index, ranks


def takepro(elem):
    return elem[1]


def sort(lis):
    '''
    lis中的每个元素是一个概率，排序之后的第一个数字是最好的回复的 id，第二个数字是第二好的回复的 id，
    返回排序之后的索引list
    '''
    temp_list = [(idx, pro) for idx, pro in enumerate(lis)]
    temp_list.sort(key=takepro, reverse=True)
    fina_list = [elem[0] for elem in temp_list]
    return fina_list


def imshow(obj, title):
    epoch = [i['epoch'] for i in obj]
    MRR = [i['MRR'] for i in obj]
    R10_1 = [i['R10_1'] for i in obj]
    R10_2 = [i['R10_2'] for i in obj]
    R10_5 = [i['R10_5'] for i in obj]
    plt.figure()
    plt.plot(epoch, MRR, label='MRR/MAP', color="#F08080")
    plt.plot(epoch, R10_1, label='R10_1/P_1', color='#DB7093')
    plt.plot(epoch, R10_2, label='R10_2', color='#1E90FF')
    plt.plot(epoch, R10_5, label='R10_5', color='#00FFFF')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.xticks(epoch)
    plt.grid(alpha=0.4, linestyle=':')
    plt.show()


def compare(lis1, lis2):
    assert len(lis1) == len(lis2)
    num = 0
    for (a, b) in zip(lis1, lis2):
        if a == b:
            num += 1
    return num / len(lis1)


def output(lis):
    lis_out = []
    for sub_lis in lis:
        for item in sub_lis:
            lis_out.append(str(item) + '\n')
        lis_out.append('\n')
    return lis_out
