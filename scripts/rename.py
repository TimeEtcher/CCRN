import os

def main (path):
    filename_list = os.listdir(path)
    """os.listdir(path) 扫描路径的文件，将文件名存入存入列表"""

    a = 5001
    for i in range(len(filename_list)):
        used_name = os.path.join(path, filename_list[i])
        x = a-5000
        new_name = os.path.join(path,str(x)+ used_name[used_name.index('.'):]) # 保留原后缀
        os.rename(used_name, new_name)
        a += 1

if __name__=='__main__':
    path=r"/media/a808/beta_2t/SR/dataset/DIV2K_decoded/DIV2K_train_HR" # 目标路径
    main(path)
