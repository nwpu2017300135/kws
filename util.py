import math
import matplotlib.pyplot as plt
def getMax(smooth_probability,h_max,j):
    max_label0 = -1
    max_label1 = -1
    for i in range(h_max,j+1):
        # print("cutcutcut",smooth_probability[i][0],smooth_probability[i][1])
        if smooth_probability[i][0] > max_label0:
            max_label0 = smooth_probability[i][0]
        if smooth_probability[i][1] > max_label1:
            max_label1 = smooth_probability[i][1]
    return max_label0,max_label1


def posteriorHandling(probability,fbankEndFrame):
    

    confidence = []

    for i in range(len(fbankEndFrame)):
        if i == 0:
            fbank_probability = probability[0:fbankEndFrame[0]]
        else:
            fbank_probability = probability[fbankEndFrame[i - 1]:fbankEndFrame[i]]
        
        #print("fbank_pro",fbank_probability)
        smooth_probability = []

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
        
        #print("smooth",smooth_probability)

        frame_confidence = []
        confidence_temp = 0

        for j in range(len(fbank_probability)):
            w_max = 100
            h_max = max(0,(j - w_max + 1))
            max_label0,max_label1 = getMax(smooth_probability,h_max,j)
            confidence_temp = (max_label0 * max_label1)
            frame_confidence.append(confidence_temp)

        confidence.append(math.sqrt(max(frame_confidence)))

        #print("fbank_pro",fbank_probability)
        #print("smooth",smooth_probability)
        #print("frame_confi",frame_confidence)
        #print("confi",confidence)
    #print("smoooth_pro",smooth_probability)
    return confidence
def plot(false_alarm_rate_list, false_reject_rate_list):

    plt.figure(figsize=(8, 4))
    plt.plot(false_alarm_rate_list, false_reject_rate_list)
    plt.xlabel('false_alarm_rate')
    plt.ylabel('false_reject_rate')
    plt.title('ROC')
    plt.savefig("new.png")
    plt.show()
def get_label_list():
    l=[]
    f = open("/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/test.scp", "r")
    temp = f.read()
    test_list = temp.split('\n')
    for example in test_list:
        if example.split('/').count("positive"):
            l.append(1)
        else :
            l.append(0)
    f.close()
#    print(l)
    return l
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
    return  false_alarm_rate_list,false_reject_rate_list




