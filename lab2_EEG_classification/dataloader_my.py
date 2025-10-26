#%%
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def read_bci_data():
    # 讀取資料檔案
    S4b_train = np.load('D:/lab2_EEG_classification/lab2_EEG_classification/data/S4b_train.npz')
    X11b_train = np.load('D:/lab2_EEG_classification/lab2_EEG_classification/data/X11b_train.npz')
    S4b_test = np.load('D:/lab2_EEG_classification/lab2_EEG_classification/data/S4b_test.npz')
    X11b_test = np.load('D:/lab2_EEG_classification/lab2_EEG_classification/data/X11b_test.npz')

    # 合併訓練和測試資料
    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    # 將標籤從 1,2 轉換為 0,1
    train_label = train_label - 1
    test_label = test_label - 1
    
    # 調整資料維度：[N, 2, 750] -> [N, 1, 2, 750]
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    # 處理 NaN 值
    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(f"訓練資料形狀: {train_data.shape}")
    print(f"訓練標籤形狀: {train_label.shape}")
    print(f"測試資料形狀: {test_data.shape}")
    print(f"測試標籤形狀: {test_label.shape}")
    print(f"標籤範圍: {np.unique(train_label)}")

    return train_data, train_label, test_data, test_label

class EEGDataset(Dataset):
    """
    EEG Dataset 類別，用於 PyTorch DataLoader
    """
    def __init__(self, data, labels):
        """
        Args:
            data: numpy array [N, 1, 2, 750]
            labels: numpy array [N]
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            data: [1, 2, 750] - EEG 訊號
            label: scalar - 類別標籤 (0 或 1)
        """
        return self.data[idx], self.labels[idx]


def get_dataloaders(batch_size=64, num_workers=0):
    """
    建立訓練和測試的 DataLoader
    
    Args:
        batch_size: batch 大小
        num_workers: 資料載入的 worker 數量
    
    Returns:
        train_loader: 訓練資料的 DataLoader
        test_loader: 測試資料的 DataLoader
    """
    # 讀取資料
    train_data, train_label, test_data, test_label = read_bci_data()
    
    # 建立 Dataset
    train_dataset = EEGDataset(train_data, train_label)
    test_dataset = EEGDataset(test_data, test_label)
    
    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader


# 測試函數
def test_dataloader():
    """
    測試 DataLoader 是否正常運作
    """
    print("\n" + "=" * 60)
    print("測試 DataLoader")
    print("=" * 60)
    
    try:
        # 建立 DataLoader
        train_loader, test_loader = get_dataloaders(batch_size=32)
        
        # 測試訓練資料
        print("\n【訓練資料】")
        for batch_idx, (data, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx + 1}:")
            print(f"  資料形狀: {data.shape}")  # [B, 1, 2, 750]
            print(f"  標籤形狀: {labels.shape}")  # [B]
            print(f"  資料範圍: [{data.min():.2f}, {data.max():.2f}]")
            print(f"  標籤範例: {labels[:10].numpy()}")
            print(f"  標籤分佈: Class 0: {(labels == 0).sum()}, Class 1: {(labels == 1).sum()}")
            break
        
        print(f"\n訓練總批次數: {len(train_loader)}")
        print(f"測試總批次數: {len(test_loader)}")
        
        # 測試測試資料
        print("\n【測試資料】")
        for batch_idx, (data, labels) in enumerate(test_loader):
            print(f"Batch {batch_idx + 1}:")
            print(f"  資料形狀: {data.shape}")
            print(f"  標籤形狀: {labels.shape}")
            print(f"  標籤範例: {labels[:10].numpy()}")
            break
        
        print("\n" + "=" * 60)
        print("✓ DataLoader 測試通過！")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ 錯誤：找不到資料檔案！")
        print(f"錯誤訊息: {e}")
        print("\n請確認以下檔案存在：")
        print("  - D:/lab2_EEG_classification/lab2_EEG_classification/data/S4b_train.npz")
        print("  - D:/lab2_EEG_classification/lab2_EEG_classification/data/X11b_train.npz")
        print("  - D:/lab2_EEG_classification/lab2_EEG_classification/data/S4b_test.npz")
        print("  - D:/lab2_EEG_classification/lab2_EEG_classification/data/X11b_test.npz")
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataloader()

# %%
