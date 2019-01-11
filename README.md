# gan-image-metrics
GANgsters

## faster-rcnn

### How to use the object recognition model

1. Created a **data/** directory within **faster-rcnn.pytorch-pytorch-1.0/**
2. Created symlink to data and put symlink in **data/**
3. Trained the vgg16 model according to README within original repo (models are saved every epoch and placed in the **models/** directory)
4. Ran this command to run object detection on all images within the **images** directory. 
```
python3 demo.py --net vgg16 --checksession 1 --checkepoch 20 --checkpoint 416 --cuda --load_dir models
```

Note: session, epoch, and checkpoint values differ based on training parameters -- so they're not general values.
