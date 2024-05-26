from sklearn.datasets import fetch_20newsgroups

def load_my_dataset_data():
    """
    Load dataset from the '20newsgroups' directory.
    """
    # Thay đổi đường dẫn tới thư mục chứa dữ liệu 20newsgroups ở đây
    DATASET_DIR = "C:/Users/NGUYENNGOC/scikit_learn_data/20news_home"

    # Load dữ liệu 20newsgroups
    data = fetch_20newsgroups(subset='all', data_home=DATASET_DIR)

    return data.data, data.target, data.target_names
