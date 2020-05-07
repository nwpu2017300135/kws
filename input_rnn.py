import fbank_reader
import numpy as np
import os
def getname(example):
    return example.split('/')[-1][:-6]

class Dataset(object):
    def __init__(self, file_list, position_list):
        self.file_list = file_list
        self.position_list = position_list
        self.position = 0
        self.length=len(file_list)
    def get_next_batch(self,batch_size):
        example=[]
        label=[]
        start=self.position%self.length
        if start==0:
            index=np.array(range(self.length))
            np.random.shuffle(index)
            self.file_list=np.array(self.file_list)[index]
        end=(self.position+batch_size)%self.length
        if start<end:
            fbank_list=self.file_list[start:end]
            self.position+=batch_size
        else :
            fbank_list=self.file_list[start:self.length]
            self.position=0
        for fileinfo in fbank_list:
            l=[]
            ll=[]
            file_path = fileinfo.split(" ")[1]
            if fileinfo.split("/").count("negative")>0:
                fbank = fbank_reader.HTKFeat_read(file_path).getall()
                for i in range(fbank.shape[0]):
                    l.append(frame_combine(i,file_path,0,fbank.shape[0]-1))
                    ll.append('2')
                example.append(l)
                label.append(ll)
            else:
                fbank = fbank_reader.HTKFeat_read(file_path).getall()
                first=self.position_list.find(getname(fileinfo))
                second=self.position_list.find("positive",first+1)
                if second!=-1:
                    second-=1
                position_str=self.position_list[first+15:second]
                keyword_position=position_str.split(" ")
                keyword_frame_position=[]
                for i in range(4):
                    frame_position=(int(keyword_position[i])-240)//160
                    if frame_position>=fbank.shape[0]:
                        frame_position=fbank.shape[0]-1
                    keyword_frame_position.append(frame_position)
                st=0
                en=keyword_frame_position[0]
                for frame in range(st,en):
                    l.append(frame_combine(frame,file_path,st,en-1))
                    ll.append('2')
                st=en
                en=keyword_frame_position[1]+1
                for frame in range(st,en):
                    l.append(frame_combine(frame,file_path,st,en-1))
                    ll.append('0')
                st=en
                en=keyword_frame_position[2]
                for frame in range(st,en):
                    l.append(frame_combine(frame,file_path,st,en-1))
                    ll.append('2')
                st=en
                en=keyword_frame_position[3]+1
                for frame in range(st,en):
                    l.append(frame_combine(frame,file_path,st,en-1))
                    ll.append('1')
                st=en
                en=fbank.shape[0]
                for frame in range(st,en):
                    l.append(frame_combine(frame,file_path,st,en-1))
                    ll.append('2')
                    # print("lshape",len(l))
                example.append(l)
                #print("example",len(example[0]))
                label.append(ll)
        print(len(example[0]))
        return example,label

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
    train=Dataset(train_list,position_data)
    test=Dataset(test_list,position_data)
    return train,test

if __name__ == '__main__':
    x,y=read_data_sets()
    temp=x.get_next_batch(16)
    xx=np.array(temp[0])
    yy=np.array(temp[1])
    print('xshape',xx.shape)
    print('yshape',yy.shape)
