from .data_provider import DataFromList, MultiProcessMapDataZMQ, BatchData, MapData
class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False):


        self._datalist = dataset.__getlist__()
        self._func = dataset.__func__()
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self.dataiter = self._batch_iter_()
        self._idx = 0
        self._drop_last = drop_last
        self._max_idx = int (len(self._datalist)/batch_size)
        if not drop_last:
            self._max_idx += 1
        
    def _batch_iter_(self):    
        ds = DataFromList(self._datalist, is_train=True, shuffle=self._shuffle)
        if self._num_workers > 1:
            dp = MultiProcessMapDataZMQ(ds, self._num_workers, self._func)
        else:
            dp = MapData(ds, self._func)
        batchp = BatchData(dp, self._batch_size)
        batchp.reset_state()
        dataiter = batchp.get_data()

        return dataiter

    def __iter__(self):
        return self

    
    def __next__(self):
        blobs = next(self.dataiter)
        self._idx += 1
        if self._idx <= self._max_idx :
            return  blobs
        else:
            self._idx = 0
            raise StopIteration    
