import h5py
import math

def count_split_size(dataset_split):
    split_size = 0
    for _ in dataset_split:
        split_size += 1
    return split_size


for languages in [('eng','ger'), ('eng','it'), ('eng','ru')]:
    # Split data
    lng1, lng2 = languages
    dataset_name = './data/wiw_data/' + lng1 + '_' +lng2 + '/wiw_' + lng1 + '_' + lng2 +'_img_pca_300_wv.h5'
    dataname = h5py.File(dataset_name, 'r')
    dataset_split = dataname['train']

    samples_size = count_split_size(dataset_split)
    train_size = int(math.ceil(0.7*samples_size))
    val_size = int(math.ceil(0.15*samples_size))
    test_size = samples_size - (train_size + val_size)
    splits_size = [train_size, val_size, test_size]

    train_data = []
    val_data = []
    test_data = []

    for idx in range(samples_size):
        tup = dataset_split['%06d' % idx]
        lng1_feats = tup[lng1 + '_feats'].value
        lng2_feats = tup[lng2 + '_feats'].value
        vis_feats = tup['vis_feats'].value
        lng1_desc = tup[lng1 + '_descriptions_feats'].value[0]
        lng2_desc = tup[lng2 + '_descriptions_feats'].value[0]

        if idx < train_size:
            train_data. append((lng1_feats, lng2_feats, vis_feats, lng1_desc, lng2_desc))
        elif idx < train_size + val_size:
            val_data. append((lng1_feats, lng2_feats, vis_feats, lng1_desc, lng2_desc))
        else:
            test_data. append((lng1_feats, lng2_feats, vis_feats, lng1_desc, lng2_desc))

    # Write data
    fname = dataset_name[:-3] + '_splitted'
    h5output = h5py.File(fname+".h5", "w")
    # The HDF5 file will contain a top-level group for each split
    train = h5output.create_group('train')
    val = h5output.create_group('val')
    test = h5output.create_group('test')
    splits = ['train', 'val','test']
    split_dims = []
    img_dims = train_data[0][2].shape[0]
    txt_dims = [train_data[0][0].shape[0], train_data[0][1].shape[0]]
    data_idx = 0
    for split in splits:
        dims = splits_size[data_idx]
        textual_features = []
        data_dim_idx = 0
        for dim_idx in range(dims):
            if split == 'train':
                container = train.create_group('%06d' % data_dim_idx)
            elif split == 'val':
                container = val.create_group('%06d' % data_dim_idx)
            else:
                container = test.create_group('%06d' % data_dim_idx)
            lng_idx = 0
            if split == 'train':
                for j, lng in enumerate(languages):
                    text_data = container.create_dataset(lng + '_feats', (txt_dims[j],), dtype='float32')
                    text_data[:] = train_data[dim_idx][j]
                    text_data = container.create_dataset(lng + '_descriptions_feats', (1,),dtype=h5py.special_dtype(vlen=str))
                    text_data[:] = train_data[dim_idx][3+j]
                    lng_idx+=1
                image_data = container.create_dataset('vis_feats', (img_dims,), dtype='float32')
                image_data[:] = train_data[dim_idx][2]
            elif split == 'val':
                for j, lng in enumerate(languages):
                    text_data = container.create_dataset(lng + '_feats', (txt_dims[j],), dtype='float32')
                    text_data[:] = val_data[dim_idx][j]
                    text_data = container.create_dataset(lng + '_descriptions_feats', (1,),dtype=h5py.special_dtype(vlen=str))
                    text_data[:] = val_data[dim_idx][3+j]
                    lng_idx+=1
                image_data = container.create_dataset('vis_feats', (img_dims,), dtype='float32')
                image_data[:] = val_data[dim_idx][2]
            else:
                for j, lng in enumerate(languages):
                    text_data = container.create_dataset(lng + '_feats', (txt_dims[j],), dtype='float32')
                    text_data[:] = test_data[dim_idx][j]
                    text_data = container.create_dataset(lng + '_descriptions_feats', (1,),dtype=h5py.special_dtype(vlen=str))
                    text_data[:] = test_data[dim_idx][3+j]
                    lng_idx+=1
                image_data = container.create_dataset('vis_feats', (img_dims,), dtype='float32')
                image_data[:] = test_data[dim_idx][2]
            data_dim_idx+=1
        data_idx+=1
    h5output.close()