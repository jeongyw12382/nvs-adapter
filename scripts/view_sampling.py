import argparse
from pathlib import Path
import os
from typing import Optional

import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import pyrootutils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sgm.util import instantiate_from_config
from sgm.data.dirdataset import load_json_data, decode_image
from sgm.geometry import make_4x4_matrix, compute_inverse_transform, make_intrinsic_matrix


class DirDatasetViewExp(Dataset):

    color_background = [255.0, 255.0, 255.0, 255.0]

    def __init__(self, ds_root_path: str, ds_list_json_path: str, num_total_views: int, num_repeat_views: int, num_query_views, resolution: int, use_relative: bool, repeat=5):
        super().__init__()

        self.ds_root_path = ds_root_path
        self.ds_list_json_path = ds_list_json_path
        self.num_total_views = num_total_views
        self.num_repeat_views = num_repeat_views
        self.num_query_views = num_query_views

        ds_list = load_json_data(ds_list_json_path)
        ds_list = list(dict.fromkeys(["_".join(ds.split("_")[:-1]) for ds in ds_list]))
        self.ds_list = [ds for ds in ds_list for _ in range(repeat)]
        self.resolution = resolution
        self.use_relative = use_relative

    def __len__(self):
        return len(self.ds_list)

    def __getitem__(self, index):
        ds_path = self.ds_list[index]

        path_list = []
        indices = np.concatenate([np.array([0]), np.random.choice(self.num_total_views - 1, self.num_query_views - 1, replace=False) + 1])

        idx = 0
        postfix = str(idx + 1).zfill(3)
        path_list.append((
            os.path.join(self.ds_root_path, ds_path + "_" + postfix, "source.png"),
            os.path.join(self.ds_root_path, ds_path + "_" + postfix, "source.npy")
        ))

        for idx in indices:
            postfix = str(idx + 1).zfill(3)
            path_list.append((
                os.path.join(self.ds_root_path, ds_path + "_" + postfix, "target.png"),
                os.path.join(self.ds_root_path, ds_path + "_" + postfix, "target.npy")
            ))

        fov_rad = np.deg2rad(49.1)  # for objaverse rendering dataset

        num_views_each = [1, self.num_query_views]

        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (self.resolution, self.resolution),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])

        rgbs, intrinsics, c2ws = [], [], []

        for png_path, npy_path in path_list:
            image = image_transform(decode_image(png_path, self.color_background))

            w2c = np.load(npy_path).astype(np.float32)
            w2c = make_4x4_matrix(torch.tensor(w2c))
            c2w = compute_inverse_transform(w2c)
            c2w[..., :3, :3] *= -1

            intrinsic = make_intrinsic_matrix(fov_rad=fov_rad, h=image.shape[1], w=image.shape[2])

            rgbs.append(image)
            intrinsics.append(intrinsic)
            c2ws.append(c2w)

        rgbs, intrinsics, c2ws = map(lambda x: torch.stack(x), (rgbs, intrinsics, c2ws))

        support_rgbs, query_rgbs = torch.split(rgbs, num_views_each)
        support_intrinsics, query_intrinsics = torch.split(intrinsics, num_views_each)
        support_c2ws, query_c2ws = torch.split(c2ws, num_views_each)

        if self.use_relative:
            inverse_support_c2ws = torch.inverse(support_c2ws)
            support_c2ws = inverse_support_c2ws @ support_c2ws
            query_c2ws = inverse_support_c2ws @ query_c2ws

        return dict(
            support_rgbs=support_rgbs,
            support_intrinsics=support_intrinsics,
            support_c2ws=support_c2ws,
            query_rgbs=query_rgbs,
            query_intrinsics=query_intrinsics,
            query_c2ws=query_c2ws
        )


class DirDataModuleViewExp(LightningDataModule):

    def __init__(
        self,
        ds_root_path: str,
        ds_list_json_path: str,
        num_total_views: int,
        batch_size: int,
        num_workers: int,
        resolution: int,
        use_relative: bool,
        num_repeat_views: int, 
        num_query_views: int
    ):
        super(DirDataModuleViewExp, self).__init__()

        self.ds_root_path = ds_root_path
        self.ds_list_json_path = ds_list_json_path
        self.num_total_views = num_total_views
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.use_relative = use_relative
        self.num_repeat_views = num_repeat_views
        self.num_query_views = num_query_views

        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: str) -> None:
        if stage == "test" or stage is None:
            self.test_dataset = DirDatasetViewExp(self.ds_root_path, self.ds_list_json_path, self.num_total_views, self.num_repeat_views, self.num_query_views,
            resolution=self.resolution, use_relative=self.use_relative)
        else:
            raise f"DirDataLoader only support test"

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


def evaluate(args):

    pl.seed_everything(args.seed)

    name = args.name
    if name is None:
        name = "noname"

    expname = os.path.splitext(os.path.basename(args.config_path))[0]

    save_dir = Path(args.logdir, expname)
    save_dir.mkdir(exist_ok=True)

    with open(args.config_path) as fp:
        config = OmegaConf.load(fp)

    for cfg_path in args.additional_configs:
        with open(cfg_path) as fp:
            config = OmegaConf.merge(config, OmegaConf.load(fp))

    model_config = config.model
    model_config.params.use_ema = args.use_ema
    model_config.params.sd_ckpt_path = None
    model_config.params.ckpt_path = args.ckpt_path

    data_config = config.data

    if args.cfg_scale is not None:
        model_config.params.sampler_config.params.guider_config.params.scale = args.cfg_scale
        cfg_scale = args.cfg_scale
    else:
        cfg_scale = model_config.params.sampler_config.params.guider_config.params.scale

    dirname = f"{name}_step_cfg_scale_{cfg_scale}_use_ema_{args.use_ema}_seed_{args.seed}"
    if args.split_idx is not None:
        dirname = dirname + "_" + f"{args.split_idx}"

    save_dir = save_dir.joinpath(dirname)
    save_dir.mkdir(exist_ok=True)

    litmodule = instantiate_from_config(model_config)
    litmodule.save_dir = save_dir

    datamodule = DirDataModuleViewExp(
        ds_root_path=args.ds_root_path,
        ds_list_json_path=args.ds_list_json_path,
        num_total_views=args.ds_num_total_views,
        num_repeat_views=args.num_repeat_views,
        num_query_views=model_config.params.network_config.params.num_query,
        batch_size=args.batch_size,
        num_workers=data_config.params.num_workers,
        resolution=data_config.params.val_config.resolution,
        use_relative=data_config.params.val_config.use_relative,
    )

    trainer = pl.Trainer(devices=1)
    trainer.test(litmodule, dataloaders=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None, required=True, help="path to config of trained model")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, required=True, help="path to checkpoint of trained model"
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="name of the visualization")
    parser.add_argument("--logdir", type=str, default="./logs_sampling", help="path to save the visualization")
    parser.add_argument("--use_ema", action="store_true", default=False, help="whether to use EMA model")
    parser.add_argument("--cfg_scale", type=float, default=None, help="scale for classifier free guidance")
    parser.add_argument("--ds_name", type=str, default="objaverse", help="the name of dataset")
    parser.add_argument("--ds_root_path", type=str, help="path to dataset for test", required=True)
    parser.add_argument("--ds_list_json_path", type=str, help="json path for list of dataset", required=True)
    parser.add_argument("--ds_num_total_views", type=int, help="number of total views per scene", required=True)
    parser.add_argument("--num_repeat_views", type=int, help="number of total views per scene", default=5)
    parser.add_argument("--split_idx", type=int, default=None, help="split index for dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for test")
    parser.add_argument("--seed", type=int, default=0, help="seed for random number generator")
    parser.add_argument("-c", "--additional_configs", nargs="*", default=list())
    args = parser.parse_args()

    print("=" * 100)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 100)

    evaluate(args)
