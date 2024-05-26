from sklearn.datasets import fetch_20newsgroups

# Tải dữ liệu từ bộ dữ liệu 20 newsgroups
data = fetch_20newsgroups()

# Lưu dữ liệu vào file văn bản
with open("20_newsgroups_data.txt", "w", encoding="utf-8") as file:
    for i in range(len(data.data)):
        file.write(data.data[i])
        file.write("\n\n")
        file.write("=" * 50)
        file.write("\n\n")

print("Dữ liệu đã được lưu vào file 20_newsgroups_data.txt")
