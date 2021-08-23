import datetime
import matplotlib.pyplot as plt
import os


def plot_loss_and_lr(train_loss, learning_rate):
    # write files
    if not os.path.exists('./log/loss_and_lr'):
        os.makedirs('./log/loss_and_lr')
    with open("./log/loss_and_lr/loss_and_lr{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), "w") as f:
        for i in range(len(train_loss)):
            f.write(str(train_loss[i]))
            f.write('\t')
            f.write(str(learning_rate[i]))
            f.write('\n')
    # plot figure
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
        fig.savefig('./log/loss_and_lr/loss_and_lr{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_map(mAP):
    # write files
    if not os.path.exists('./log/mAP'):
        os.makedirs('./log/mAP')
    with open("./log/mAP/mAP{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), "w") as f:
        for map in mAP:
            f.write(str(map))
            f.write('\n')
    # plot figure
    try:
        x = list(range(len(mAP)))
        plt.plot(x, mAP, label='mAp')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.title('Eval mAP')
        plt.xlim(0, len(mAP))
        plt.legend(loc='best')
        plt.savefig('./log/mAP/mAP{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("successful save mAP curve!")
    except Exception as e:
        print(e)
