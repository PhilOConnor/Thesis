import matplotlib.pyplot as plt
import os
import pandas as pd


file_list = os.listdir('../Outputs/csv/')

for file in file_list:
    try:
        output_filename = os.path.splitext(file)
        df = pd.read_excel(os.path.join('../Outputs/csv/', file), sheet_name='model history')
        plt.subplot(1,2,1)
        plt.plot(df['accuracy'])
        plt.plot(df['val_accuracy'])
        plt.axvline(x = 1000, color = 'b',linestyle='dashed', label = 'fine-tuning')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.ylim(0, 1)
        plt.legend(['train', 'val','fine-tuning'], loc='lower left')
        plt.subplot(1,2,2)
        # summarize history for loss
        plt.plot(df['loss'])
        plt.plot(df['val_loss'])
        plt.axvline(x = 1000, color = 'b', linestyle='dashed', label = 'fine-tuning')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.ylim(0, 3)
        plt.legend(['train', 'val','fine-tuning'], loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join('../Outputs/figures/graphs2/', f"{output_filename[0]}.jpg"))
        plt.clf()
    except:
        pass
