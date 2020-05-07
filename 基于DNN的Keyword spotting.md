## 基于DNN的Keyword spotting

本实验实现基于DNN的关键词检出。

参考论文：SMALL-FOOTPRINT KEYWORD SPOTTING USING DEEP NEURAL NETWORKS

### 试验方法

将音频进行fbank处理后，需要进行拼帧将前方30帧和后方10帧拼在一起作为一个数据。

之后构建网络，采用RELU作为激活函数，输入大小为拼帧后一个数据的维度（1640），输出为类别数（3），中间为全连接网络，这里采用三层256的网络。

得到输出后要进行后处理。首先是smooth处理，直观上来说就是对每一个帧的输出结果转化为一个窗口内输出的平均值。得到了smooth之后的输出后，对于j帧，将一个窗口内的最大0类输出和1类输出相乘就得到了j的confidence，一段音频的confidence超过阈值就视为监测到关键词，否则视为未检测到。通过遍历阈值得到ROC曲线，进而评估性能，确定最佳阈值。

### 实验数据

实验数据为经过fbank处理后的音频，分为train和test，对于含有关键词的音频（即正例）将对应关键词的起始位置记录在positivekeywordposition中。

#### 数据处理

对于导入的数据，首先是需要拼帧，拼帧实现如下：

```python
def frame_combine(frame, file_path, start, end):
    fbank = fbank_reader.HTKFeat_read(file_path).getall()
#长度小于10+30+1
    if end - start + 1 < 41:
        if frame - start <= 30 and end - frame <= 10: #前后均不足
            frame_to_combine = []
            front_rest = 30 - (frame - start)
            back_rest = 10 - (end - frame)
            for i in range(front_rest):
                frame_to_combine.append(fbank[start].tolist())
            for i in range(start, end + 1):
                frame_to_combine.append(fbank[i].tolist())
            for i in range(back_rest):
                frame_to_combine.append(fbank[end].tolist())

        elif end - frame >= 10:#后足只补前
            frame_to_combine = []
            front_rest = 30 - (frame - start)
            for i in range(front_rest):
                frame_to_combine.append(fbank[start].tolist())
            for i in range(start, frame+11):
                frame_to_combine.append(fbank[i].tolist())

        else: #前足
            frame_to_combine = []
            back_rest = 10 - (end - frame)
            for i in range(frame - 30, end + 1):
                frame_to_combine.append(fbank[i].tolist())
            for i in range(back_rest):
                frame_to_combine.append(fbank[end].tolist())
        combined = np.array(frame_to_combine).reshape(-1)

    else:#长度超过41
        if frame - start >= 30 and end - frame >= 10:#前后均够
            frame_to_combine = fbank[frame - 30: frame + 11]
            combined = frame_to_combine.reshape(-1)

        elif frame - start < 30:
            frame_to_combine = fbank[start: start+41]
            combined = frame_to_combine.reshape(-1)

        else:
            frame_to_combine = fbank[end - 40: end+1]
            combined = frame_to_combine.reshape(-1)

    return combined.tolist()


```

生成了数据之后就要对数据打标记。对于 `hello` 帧我们标记`0`，对于`小瓜`帧我们标记`1`对于非关键词我们标价`2`

对于postive部分：

```python
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
```

对negative部分：

```python
  for frame in range(frame_number):
                    self.example.append(frame_combine(frame, file_path, 0, frame_number - 1))
                    self.labels.append('2')
                    self.num_examples += 1
```

#### 数据导入

由于训练集数据量大，所以采用分批导入，一次导入10条音频，用尽后再导入，训练时也用gennertor导入。

而测试集则一次全部导入。

### 网络部分

采用六层个12的全连接层，dropout使30%的神经元失活，relu作为激活函数，最后网络输出三个类别的低分计算softmax作为最终输出。

损失函数为交叉熵，优化器选择SGD，学习率选择0.001.

然后进行训练。

代码如下：

```python
model = Sequential()
model.add(Dense(512,input_shape=(1640,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.SGD(lr=0.001, momentum=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.fit_generator(trans(train,batch_size),
              epochs=epochs,
              validation_data=(x_test, y_test),
              steps_per_epoch=13,
              verbose=2,
              workers=0)
```

### 后处理部分

训练模型之后，使用模型对test数据进行预测。

```probability=model.predict(x_test,verbose=0,batch_size=batch_size)```

得到结果需要进行后处理来评估性能。

#### smooth

首先是smooth处理。



```python
for j in range(len(fbank_probability)):
            sum_label0 = 0
            sum_label1 = 0
            sum_label2 = 0
            w_smooth = 30
            h_smooth = max(0,(j - w_smooth + 1))

            division = j - h_smooth + 1
            for temp in range(h_smooth,j+1):
                sum_label0 += fbank_probability[temp][0]
                sum_label1 += fbank_probability[temp][1]
                sum_label2 += fbank_probability[temp][2]
            smooth_probability.append([sum_label0/division,sum_label1/division,sum_label2/division])
```

#### confidence

然后是计算每一帧的confidence

```python
for j in range(len(fbank_probability)):
            w_max = 100
            h_max = max(0,(j - w_max + 1))
            max_label0,max_label1 = getMax(smooth_probability,h_max,j)
            confidence_temp = (max_label0 * max_label1)
            frame_confidence.append(confidence_temp)
```

#### 评估

遍历阈值，绘制ROC

```python
def do_eval(confidence):
    label=get_label_list()
    threshold_part = 10000
    false_alarm_rate_list=[]
    false_reject_rate_list=[]

    for i in range(1,threshold_part):
        threshold=float(i)/threshold_part
        true_alarm = true_reject = false_reject = false_alarm = 0
        for j in range(len(confidence)):
            if confidence[j]<threshold:
                if label[j]==0:
                    true_reject+=1
                else:
                    false_reject+=1
            else :
                if label[j]==0:
                    false_alarm+=1
                else :
                    true_alarm+=1
        if false_reject + true_reject == 0 or false_alarm + true_alarm == 0:
            continue
        false_alarm_rate = float(false_alarm) / (false_alarm + true_alarm)
        false_reject_rate = float(false_reject) / (false_reject + true_reject)
        false_alarm_rate_list.append(false_alarm_rate)
        false_reject_rate_list.append(false_reject_rate)
    plot(false_alarm_rate_list, false_reject_rate_list)
    print(false_alarm_rate_list[0::100])

```

### 结果分析

