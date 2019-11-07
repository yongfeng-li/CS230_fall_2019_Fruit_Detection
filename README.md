CS230 project - Fruit Detection for group of Anna-Yongfeng-Ajay (AYA)

If want to run on GPU>2, need to change code in mrcnn/parallel_model.py

class ParallelModel(KM.Model):
    def __init__(self, keras_model, gpu_count):
        """Class constructor.
        keras_model: The Keras model to parallelize
        gpu_count: Number of GPUs. Must be > 1
        """
        #ADD THIS LINE---
        super(ParallelModel, self).__init__()
        #END
        self.inner_model = keras_model
        self.gpu_count = gpu_count
        merged_outputs = self.make_parallel()
        super(ParallelModel, self).__init__(inputs=self.inner_model.inputs,
                                            outputs=merged_outputs)

20191102:Update---

uploaded the "MaskRCNN train" & coco data selection file. Anna 

------------------------------------------------------------------------------------

Hi GUYS, ADDED some AWS tips, saw someone asking this on piazza

1) install aws-cli :https://docs.aws.amazon.com/zh_cn/cli/latest/userguide/cli-chap-install.html
2) setup awc config( need key) : https://docs.aws.amazon.com/zh_cn/cli/latest/userguide/cli-chap-configure.html
- access key ID
- secret access key
- region
- output format

all set use python. (  need to create local fold ) 
