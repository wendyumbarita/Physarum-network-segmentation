# Dataset Setup

Only include datasets you have permission to distribute. Dataset files are excluded by `.gitignore`.

Run from the repository root directory.

## Full-image resized training

Uses `SLIME_DATA_DIR` (default: `./Data`):

```text
Data/
├── images/
│   └── sample.jpg
└── network_filled/
    └── sample_Network.png
```

The full-image script expects masks to match image stems with the `_Network.png` suffix.

## Patch-based training

Uses `SLIME_BASE_DIR` (default: repository root):

```text
Data/
├── blue_BG/
│   ├── images/
│   └── network/
├── yellow_BG/
│   ├── images/
│   └── network/
└── nice_temp/
    ├── images/
    └── network/
```

Every image needs a matching `<stem>_Network.png` mask in that subset's `network/` directory. Folder names reflect the original training code; adapt locally if your approved data is organized differently.


The base directory for patch training should contain the `Data/` folder, and outputs will also be written beneath that base. For a read-only dataset, copy or adapt the output paths first.
