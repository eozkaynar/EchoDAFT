import torchvision
import torch
import pandas as pd
import numpy as np
import os

import echodaft
from echodaft.utils import compute_discs_area

class EchoDAFT(torchvision.datasets.VisionDataset):
    def __init__(self, root, split, length=32, period=2, clips=1,
                 video_mean=0., video_std=1.,
                 tab_mean=0., tab_std=1.,
                 pad=None, transform=None):
        super(EchoDAFT, self).__init__(root)

        self.split          = split.upper()
        self.length         = length
        self.period         = period
        self.clips          = clips
        self.video_mean     = video_mean
        self.video_std      = video_std
        self.tab_mean       = tab_mean
        self.tab_std        = tab_std
        self.transform      = transform
        self.pad            = pad
        self.samples        = []
        self.fnames         = []    

        # Load Tabular Data
        # Load and process Simpson's disc data
        self.simpson = pd.read_csv(os.path.join(self.root,"Tabular", "simpsons_ed_es.csv"))
        self.simpson = self.simpson[self.simpson["Phase"].isin(["ED", "ES"])]
        
        # Filter by dataset split (train/val/test/all)
        if self.split != "ALL":
            self.simpson = self.simpson[self.simpson["Split"].astype(str).str.upper() == self.split]

        self.disc_df = echodaft.utils.compute_discs_length(self.simpson)
        self.area_df = echodaft.utils.compute_discs_area(self.disc_df)


        # Fast lookup dicts     
        self.simpson_dict   = self._build_lookup_dict(self.simpson, "Length", type_filter="Major Axis")
        self.area_dict      = self._build_lookup_dict(self.area_df, "LV_Area_Simpson", type_filter=None, agg_func="mean")
        
        for filename, group in self.simpson.groupby('Filename'):
            ef_vals = group['EF'].dropna().unique()
            if len(ef_vals) == 0:
                    continue
            ef = float(ef_vals[0])
            self.samples.append((filename, ef))
            self.fnames.append(filename)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        filename, ef = self.samples[index]
        video = os.path.join(self.root, "Videos", filename)
        
        # Load video into np.array uint8 -> float32, shape (C, F, H, W)
        video = echodaft.utils.loadvideo(video).astype(np.float32)
        c, f, h, w = video.shape  # pylint: disable=E0633

        # Apply normalization
        if isinstance(self.video_mean, (float, int)):
            video -= self.video_mean
        else:
            video -= self.video_mean.reshape(3, 1, 1, 1)

        if isinstance(self.video_std, (float, int)):
            video /= self.video_std
        else:
            video /= self.video_std.reshape(3, 1, 1, 1)


        # Pad video with frames filled with zeros if too short
        if f < self.length * self.period:
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, self.length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        # Choose clip start(s)
        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (self.length - 1) * self.period)

        else:
            # Take random clips from video
            start = np.random.choice(f - (self.length - 1) * self.period, self.clips)

        # Select clips from video
        video = tuple(video[:, s + self.period * np.arange(self.length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]

        # Compute tabular features

        major_ed = self.simpson_dict.get((filename, "ED"), 0)
        major_es = self.simpson_dict.get((filename, "ES"), 0)
    
        disc_ed = self.area_dict.get((filename, "ED"), 0)
        disc_es = self.area_dict.get((filename, "ES"), 0)
        
        area_diff_disc  = disc_ed - disc_es
        major_diff      = major_ed - major_es

        FAC   = area_diff_disc / disc_ed if disc_ed != 0 else 0
        DFC   = major_diff / major_ed if major_ed != 0 else 0

        tabular = torch.tensor([
            major_ed, major_es, 
            disc_ed, disc_es, 
            area_diff_disc, major_diff,
            FAC, DFC
            ], dtype=torch.float32)

        # Normalize tabular features
        if not torch.is_tensor(self.tab_mean):
            tab_mean = torch.tensor(self.tab_mean, dtype=torch.float32)
            tab_std = torch.tensor(self.tab_std, dtype=torch.float32)
        else:
            tab_mean = self.tab_mean
            tab_std = self.tab_std

        tabular = (tabular - tab_mean) / (tab_std + 1e-6)


        y       = torch.tensor(ef/100.0, dtype=torch.float32)
        video   = torch.from_numpy(video).float()

        return video, tabular, y, filename
    
    def _build_lookup_dict(self, df, value_column, type_filter=None, agg_func="mean"):
        if type_filter:
            df = df[df["Type"] == type_filter]

        # (Filename, Phase) başına tek satıra indir
        df_grouped = df.groupby(["Filename", "Phase"], as_index=False)[value_column].agg(agg_func)

        return {
            (row["Filename"], row["Phase"]): row[value_column]
            for _, row in df_grouped.iterrows()
        }
