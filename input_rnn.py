import fbank_reader
import numpy as np
import os
def getname(example):
    return example.split('/')[-1][:-6]
class TestSet(object):
    def __init__(self, exampls, labels, num_examples, fbank_end_frame):
        self._exampls = exampls
        self._labels = labels
        self._index_in_epochs = 0
        self.num_examples = num_examples
        self.fbank_end_frame = fbank_end_frame

    def next_batch(self, batch_size):
        start = self._index_in_epochs

        if start + batch_size > self.num_examples:
            self._index_in_epochs = self.num_examples
            end = self._index_in_epochs
            return self._exampls[start:end], self._labels[start:end]
        else:
            self._index_in_epochs += batch_size
            end = self._index_in_epochs
            return self._exampls[start:end], self._labels[start:end]


class TrainSet(object):
    def __init__(self, examples_list, position_data):
        self.examples_list = examples_list
        self.position_data = position_data
        self.fbank_position = 0
        self.index_in_epochs = 0
        self.example = []
        self.labels = []
        self.num_examples = 0

    def read_train_set(self):
        self.example = []
        self.labels = []
        self.num_examples = 0
        step_length = 10
        start = self.fbank_position % len(self.examples_list)
        end = (self.fbank_position + step_length) % len(self.examples_list)
        if start < end:
            fbank_list = self.examples_list[start: end]
            self.fbank_position += step_length

        else:
            fbank_list = self.examples_list[start: len(self.examples_list)]
            self.fbank_position = 0
            index = np.arange(len(self.examples_list))
            np.random.shuffle(index)
            self.examples_list = np.array(self.examples_list)[index]

        for example in fbank_list:
            if example == '':
                continue
            file_path = example.split(" ")[1]
            if example.split('/').count("positive")>0:
                start = self.position_data.find(getname(example))
                end = self.position_data.find("positive", start + 1)
                if end != -1:
                    position_str = self.position_data[start + 15: end - 1]
                else:
                    position_str = self.position_data[start + 15: end]

                # start and end position of "hello" & start and end position of "xiao gua"
                keyword_position = position_str.split(" ")

                file_path = example.split(" ")[1]

                keyword_frame_position = []
                for i in range(4):
                    fbank = fbank_reader.HTKFeat_read(file_path).getall()
                    length = fbank.shape[0]
                    frame_position =(int(keyword_position[i])-240) // 160
                    if frame_position >= length:
                        frame_position = length - 1
                    keyword_frame_position.append(frame_position)

                #print (example)


                for frame in range(keyword_frame_position[0], keyword_frame_position[1] + 1):
                    self.example.append(
                        frame_combine(frame, file_path, keyword_frame_position[0], keyword_frame_position[1]))
                    self.labels.append('0')
                    self.num_examples += 1
                for frame in range(keyword_frame_position[2], keyword_frame_position[3] + 1):
                    self.example.append(
                        frame_combine(frame, file_path, keyword_frame_position[2], keyword_frame_position[3]))
                    self.labels.append('1')
                    self.num_examples += 1

            else:
                file_path =example.split(" ")[1]

                fbank = fbank_reader.HTKFeat_read(file_path).getall()
                frame_number = fbank.shape[0]

                #print (example)








                for frame in range(frame_number):
                    self.example.append(frame_combine(frame, file_path, 0, frame_number - 1))
                    self.labels.append('2')
                    self.num_examples += 1
#                    print(self.example.shape)
    def next_batch(self, batch_size):
        start = self.index_in_epochs

        if start == 0:
            self.read_train_set()
            index0 = np.arange(self.num_examples)
            np.random.shuffle(index0)
            self.example = np.array(self.example)[index0]
            self.labels = np.array(self.labels)[index0]

        if start + batch_size > self.num_examples:
            examples_rest_part = self.example[start: self.num_examples]
            labels_rest_part = self.labels[start: self.num_examples]
            self.index_in_epochs = 0
            return examples_rest_part, labels_rest_part

        else:
            self.index_in_epochs += batch_size
            end = self.index_in_epochs
            return self.example[start:end], self.labels[start:end]


def frame_combine(frame, file_path, start, end):
    fbank = fbank_reader.HTKFeat_read(file_path).getall()

    if end - start + 1 < 41:
        if frame - start <= 30 and end - frame <= 10:
            frame_to_combine = []
            front_rest = 30 - (frame - start)
            back_rest = 10 - (end - frame)
            for i in range(front_rest):
                frame_to_combine.append(fbank[start].tolist())
            for i in range(start, end + 1):
                frame_to_combine.append(fbank[i].tolist())
            for i in range(back_rest):
                frame_to_combine.append(fbank[end].tolist())

        elif end - frame >= 10:
            frame_to_combine = []
            front_rest = 30 - (frame - start)
            for i in range(front_rest):
                frame_to_combine.append(fbank[start].tolist())
            for i in range(start, frame+11):
                frame_to_combine.append(fbank[i].tolist())

        else:
            frame_to_combine = []
            back_rest = 10 - (end - frame)
            for i in range(frame - 30, end + 1):
                frame_to_combine.append(fbank[i].tolist())
            for i in range(back_rest):
                frame_to_combine.append(fbank[end].tolist())
        combined = np.array(frame_to_combine).reshape(-1)

    else:
        if frame - start >= 30 and end - frame >= 10:
            frame_to_combine = fbank[frame - 30: frame + 11]
            combined = frame_to_combine.reshape(-1)

        elif frame - start < 30:
            frame_to_combine = fbank[start: start+41]
            combined = frame_to_combine.reshape(-1)

        else:
            frame_to_combine = fbank[end - 40: end+1]
            combined = frame_to_combine.reshape(-1)

    return combined.tolist()


def read_data_sets():
    f = open("/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/positiveKeywordPosition.txt", "r")
    position_data = f.read()
    f.close()

    f = open("/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/train.scp", "r")
    temp = f.read()
    train_list = temp.split('\n')[0:100]
    f.close()

    f = open("/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/test.scp", "r")
    temp = f.read()
    test_list = temp.split('\n')[0:100]
    f.close()

    test_examples = []
    test_labels = []
    test_length = []
    test_num = 0

    for example in test_list:
        if example == '':
            continue
        if example.split('/').count("positive")<1:
            continue
        start = position_data.find(getname(example))
        end = position_data.find("positive", start + 1)
        if end != -1:
            position_str = position_data[start + 15: end - 1]
        else:
            position_str = position_data[start + 15: end]

        # start and end position of "hello" & start and end position of "xiao gua"
        keyword_position = position_str.split(" ")

        file_path = example.split(" ")[1]

        keyword_frame_position = []
        for i in range(4):
            fbank = fbank_reader.HTKFeat_read(file_path).getall()
            length = fbank.shape[0]
            frame_position = int(keyword_position[i]) // 160
            if frame_position >= length:
                frame_position = length - 1
            keyword_frame_position.append(frame_position)

        test_length.append(keyword_frame_position[1] - keyword_frame_position[0] + 1 +
                                    keyword_frame_position[3] - keyword_frame_position[2] + 1)

#        print( example)
        for frame in range(keyword_frame_position[0], keyword_frame_position[1] + 1):
            test_examples.append(frame_combine(frame, file_path, keyword_frame_position[0], keyword_frame_position[1]))
            test_labels.append('0')
            test_num += 1
        for frame in range(keyword_frame_position[2], keyword_frame_position[3] + 1):
            test_examples.append(frame_combine(frame, file_path, keyword_frame_position[2], keyword_frame_position[3]))
            test_labels.append('1')
            test_num += 1
#            print(np.array(test_examples).shape)
    for example in test_list:
        if example == '':
            continue
        if example.split('/').count("negative")<1:
            continue
        file_path=example.split(" ")[1]

        fbank = fbank_reader.HTKFeat_read(file_path).getall()
        frame_number = fbank.shape[0]
        test_length.append(frame_number)
#        print (example)
        for frame in range(frame_number):
            test_examples.append(frame_combine(frame, file_path, 0, frame_number - 1))
            test_labels.append('2')
            test_num += 1
    fbank_end_frame = []
    for i in range(len(test_length)):
        fbank_end_frame.append(sum(test_length[0: i+1]))

    index = np.arange(len(train_list))
    np.random.shuffle(index)
    train_list = np.array(train_list)[index]

    train = TrainSet(train_list, position_data)
    test = TestSet(test_examples, test_labels, test_num, fbank_end_frame)

    return train, test
if __name__ == '__main__':
    x,y=read_data_sets()
    temp=x.next_batch(16)
    xx=np.array(temp[0])
    yy=np.array(temp[1])
    print('xshape',xx.shape)
    print('yshape',yy.shape)
