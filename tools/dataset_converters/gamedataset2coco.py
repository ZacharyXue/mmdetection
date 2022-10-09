# 讯飞 X光安检图像检测数据集
import os
import mmcv
import random
import argparse
import os.path as osp
import xml.dom.minidom

# 数据集结构为：
#   gamedaset
#       |-- train
#           |-- domain1
#               |-- XML
#                   |-- ...xml
#               |-- ...jpg
#           |-- domain2
#           |-- domain3
#       |-- test
#           |-- test1
# 这个结构和 coco数据集中的结构不一样，所以说需要重写 coco数据集
# 导入，或者修改数据集数据位置，这里还是想进行数据集导入的修改，
# 因为不知道 domain 实际的含义，并且三个文件中文件名有重叠，这里
# 想的是在输入路径后循环加入 domainX 来遍历数据。
# 
# 写完 json 文件才发现这个数据集并没有区分训练集和验证集，需要
# 自己划分


category2label = {
    'USBFlashDisk': 0,
    'scissors': 1,
    'knife':2,
    'plasticBottleWithaNozzle':3,
    'seal': 4,
    'lighter': 5,
    'pressure': 6,
    'battery': 7
}


def val_select(data, ratio=0.2):
    """
    从输入中选取一定数量的样本作为验证集，
    返回验证集的集合，不改变输入。
    """
    data_len = len(data)
    val_len = int(data_len * ratio)
   
    res = [i for i in range(data_len)]
    random.shuffle(res)

    return set([data[i] for i in res[:val_len]])

def convert_annotation(basedir):  
    """
    input: 数据集地址，不对domain[1-3]进行区分
    output: train.json val.json
    """
    train_json = dict()
    train_img_id = 0
    train_ann_id = 0
    train_json['images'] = []
    train_json['annotations'] = []
    train_json['categories'] = [
        {'supercategory': key, 'id': value, 'name': key}
        for key, value in category2label.items()
    ]

    val_json = dict()
    val_img_id = 0
    val_ann_id = 0
    val_json['images'] = []
    val_json['annotations'] = []
    val_json['categories'] = [
        {'supercategory': key, 'id': value, 'name': key}
        for key, value in category2label.items()
    ]

    # 遍历三个文件夹
    for num_domain in ['1', '2', '3']:
        domain = f"domain{num_domain}"
        pathlist = os.listdir(osp.join(basedir, domain, "XML"))
        val_set = val_select(pathlist)
        # 遍历 XML 文件内所有 XML 文件
        for xml_dir in pathlist:
            path = osp.join(basedir, domain, "XML", xml_dir)
            dom = xml.dom.minidom.parse(path)
            root = dom.documentElement
            filename = root.getElementsByTagName('filename')[0].childNodes[0].data
            image = mmcv.imread(osp.join(basedir, filename))
            if xml_dir not in val_set:
                train_json['images'].append({
                'file_name': filename,
                'width': image.shape[1],
                'height': image.shape[0],
                'id': train_img_id
                })
            else:
                val_json['images'].append({
                'file_name': filename,
                'width': image.shape[1],
                'height': image.shape[0],
                'id': val_img_id
                })
            for object in root.getElementsByTagName('object'):
                category = object.getElementsByTagName("name")[0].childNodes[0].data

                x_min = int(object.getElementsByTagName("bndbox")[0].getElementsByTagName('xmin')[0].firstChild.data)
                y_min = int(object.getElementsByTagName("bndbox")[0].getElementsByTagName('ymin')[0].firstChild.data)
                x_max = int(object.getElementsByTagName("bndbox")[0].getElementsByTagName('xmax')[0].firstChild.data)
                y_max = int(object.getElementsByTagName("bndbox")[0].getElementsByTagName('ymax')[0].firstChild.data)
                width = max(x_max - x_min, 0)
                height = max(y_max - y_min, 0)

                if xml_dir not in val_set:
                    train_json['annotations'].append({
                    'iscrowd': 0,
                    'image_id': train_img_id,
                    'bbox': [x_min, y_min, width, height],
                    'area': float(width * height),
                    'segmentation': [[]],
                    'category_id':category2label[category],
                    'id': train_ann_id
                    })
                    train_ann_id += 1
                else:
                    val_json['annotations'].append({
                    'iscrowd': 0,
                    'image_id': val_img_id,
                    'bbox': [x_min, y_min, width, height],
                    'area': float(width * height),
                    'segmentation': [[]],
                    'category_id':category2label[category],
                    'id': val_ann_id
                    })
                    val_ann_id += 1
            if xml_dir not in val_set:
                train_img_id += 1
            else:
                val_img_id += 1

    mmcv.dump(train_json, f'{osp.join(basedir, "train")}.json')
    mmcv.dump(val_json, f'{osp.join(basedir, "val")}.json')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Gamedataset to COCO format')
    parser.add_argument('--path', help='Gamedataset data path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    convert_annotation(osp.join(args.path, 'train'))
    # convert_annotation(osp.join(args.path, 'test/test1'))


if __name__ == '__main__':
    main()