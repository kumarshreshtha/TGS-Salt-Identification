from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import metrics



class OutputObserver(Callback):

    def __init__(self,x_val,v_labels):
        self.x_val=x_val
        self.v_labels = v_labels

    def on_epoch_end(self,epoch,logs={}):

        if epoch%3 == 0:
            predictions =  self.model.predict([self.x_val,np.expand_dims(self.x_val[...,0],axis=-1)],batch_size=1,verbose=True)
            print(np.histogram(predictions))
            fig = plt.figure(figsize = (10,40))
            plt.gray()
            plt.title('Epoch-'+str(epoch))
            rndperm = np.random.RandomState(seed=1618).permutation(predictions.shape[0])
            print(self.x_val.shape)
            print(self.v_labels.shape)
            print(predictions.shape)
            for i in range(0,100,5):
                ax = fig.add_subplot(20,5,i+1)
                image = self.x_val[rndperm[i],...,0].astype(np.uint8)
                ax.imshow(image)
                mask = (self.v_labels)[rndperm[i],...,0]
                ax = fig.add_subplot(20,5,i+2)
                ax.imshow(mask)
                predict1 = (predictions[rndperm[i],...,0]>0.5).astype(np.float64)
                ax = fig.add_subplot(20,5,i+3)
                ax.imshow(predict1)
                predict2 = (predictions[rndperm[i],...,0]>0.75).astype(np.float64)
                ax = fig.add_subplot(20,5,i+4)
                ax.imshow(predict2)
                predict3 = (predictions[rndperm[i],...,0]>0.9).astype(np.float64)
                ax = fig.add_subplot(20,5,i+5)
                ax.imshow(predict3)
            plt.show()
            thres = np.linspace(0.25, 0.75, 20)
            thres_ioc = [metrics.iou_metric_batch(self.v_labels, np.int32(predictions > t)) for t in thres]
            plt.plot(thres, thres_ioc)
            plt.show()
            best_thres = thres[np.argmax(thres_ioc)]
            print('best threshold:',best_thres,'max threshold value:' ,max(thres_ioc))