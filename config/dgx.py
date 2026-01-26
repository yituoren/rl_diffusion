import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def compressibility():
    config = base.get_config()

    # config.pretrained.model = "CompVis/stable-diffusion-v1-4"

    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    # this corresponds to (8 * 4) / (4 * 2) = 4 gradient updates per epoch.
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "jpeg_compressibility"

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }

    return config


def compressibility_and_classifier():
    config = compressibility()
    config.reward_fn = "jpeg_and_classifier"
    
    # dataset
    config.dataset = ml_collections.ConfigDict()
    config.dataset.root = "/cephfs/shared/data/cifar10"
    config.dataset.num_workers = 4
    
    config.logdir = "/cephfs/zhaorui/rl_diffusion_logs"
    
    config.sample.guidance_scale = 1.0
    config.train.cfg = False
    
    config.classifier_checkpoint = "/cephfs/shared/ckpt/cifar10_classifier/classifier_best.pth"
    
    config.classifier = {
        "model_type": "resnet",
        "resnet_arch": "resnet18",
        "pretrained": True,
        "freeze_backbone": False,
        "num_classes": 10,
        "in_channels": 3,
        "hidden_dims": [64, 128, 256],
        "fc_hidden": 256,
        "dropout": 0.3
    }
    
    return config


def incompressibility():
    config = compressibility()
    config.reward_fn = "jpeg_incompressibility"
    return config


def aesthetic():
    config = compressibility()
    config.num_epochs = 200
    config.reward_fn = "aesthetic_score"

    # this reward is a bit harder to optimize, so I used 2 gradient updates per epoch.
    config.train.gradient_accumulation_steps = 4

    config.prompt_fn = "simple_animals"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    return config


def prompt_image_alignment():
    config = compressibility()

    config.num_epochs = 200
    # for this experiment, I reserved 2 GPUs for LLaVA inference so only 6 could be used for DDPO. the total number of
    # samples per epoch is 8 * 6 * 6 = 288.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 6

    # again, this one is harder to optimize, so I used (8 * 6) / (4 * 6) = 2 gradient updates per epoch.
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 6

    # prompting
    config.prompt_fn = "nouns_activities"
    config.prompt_fn_kwargs = {
        "nouns_file": "simple_animals.txt",
        "activities_file": "activities.txt",
    }

    # rewards
    config.reward_fn = "llava_bertscore"

    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }

    return config


def get_config(name):
    return globals()[name]()
