import numpy as np
import pandas as pd
from pathlib import Path
import pickle


from load_data import generate_overlapping_windows
from format_predictions import get_predictions_indexes
import tensorflow.keras.backend as K
import tensorflow.keras as kr






## Save result if wanted:




class ExtendedRippleDetection(object):
    """docstring for ExtendedRippleDetection."""
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False):
        super(ExtendedRippleDetection, self).__init__()
        self.active_session_folder = None
        self.active_session_eeg_data_filepath = None
        self.loaded_eeg_data = None
        self.out_all_ripple_results = None
        self.ripple_df = None

        print("Loading CNN model...", end=" ")
        self.optimizer = kr.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)
        # relative:
        # model_path = "../../model"
        model_path = r"C:\Users\pho\repos\cnn-ripple\model"
        self.model = kr.models.load_model(model_path, compile=False)
        self.model.compile(loss="binary_crossentropy", optimizer=self.optimizer)
        print("Done!")


    def compute(self, active_session_folder=Path('/content/drive/Shareddrives/Diba Lab Data/KDIBA/gor01/one/2006-6-08_14-26-15'), numchannel = 96,
            srLfp = 1250, downsampled_fs = 1250, 
            overlapping = True, window_size = 0.0128, stride = 0.0064, # window parameters
            ripple_detection_threshold=0.7,
            active_shank_channels_lists = [[72,73,74,75,76,77,78,79], [81,82,83,84,85,86,87,88]]
            ):
        self.active_session_folder = active_session_folder
        self.loaded_eeg_data, self.active_session_eeg_data_filepath, self.active_session_folder = self.load_eeg_data(active_session_folder=active_session_folder, numchannel=numchannel)
        out_all_ripple_results = self.compute_ripples(self.model, self.loaded_eeg_data, srLfp=srLfp, downsampled_fs=downsampled_fs, overlapping=overlapping, window_size=window_size, stride=stride, ripple_detection_threshold=ripple_detection_threshold, active_shank_channels_lists=active_shank_channels_lists, out_all_ripple_results=None)
        self.out_all_ripple_results = out_all_ripple_results

        with open('out_all_ripple_results.pkl', 'wb') as f:
            pickle.dump(out_all_ripple_results, f)
        flattened_pred_ripple_start_stop_times = np.vstack([a_result['pred_times'] for a_result in out_all_ripple_results['results'].values()])
        print(f'flattened_pred_ripple_start_stop_times: {np.shape(flattened_pred_ripple_start_stop_times)}') # (6498, 2)
        ripple_df = pd.DataFrame({'start':flattened_pred_ripple_start_stop_times[:,0], 'stop': flattened_pred_ripple_start_stop_times[:,1]})
        self.ripple_df = ripple_df
        print(f'ripple_df: {ripple_df}')
        ripple_df.to_csv(self.predicted_ripples_dataframe_save_filepath)
        return ripple_df, out_all_ripple_results

    @property
    def predicted_ripples_dataframe_save_filepath(self):
        """The predicted_ripples_dataframe_save_filepath property."""
        return self.active_session_folder.joinpath('pred_ripples.csv')

    @classmethod
    def readmulti(cls, fname, numchannel:int, chselect=None, *args):
        """ reads multi-channel recording file to a matrix
        % 
        % function [eeg] = function readmulti(fname,numchannel,chselect)
        % last argument is optional (if omitted, it will read all the 
        % channels

        function [eeg] = readmulti(fname,numchannel,chselect,subtract_channel)

        if nargin == 2
        datafile = fopen(fname,'r');
        eeg = fread(datafile,[numchannel,inf],'int16');
        fclose(datafile);
        eeg = eeg';
        return
        end

        if nargin == 3

        % the real buffer will be buffersize * numch * 2 bytes
        % (short = 2bytes)
        
        buffersize = 4096;
        
        % get file size, and calculate the number of samples per channel
        fileinfo = dir(fname);
        numel = ceil(fileinfo(1).bytes / 2 / numchannel);
        
        datafile = fopen(fname,'r');
        
        mmm = sprintf('%d elements',numel);
        %  disp(mmm);  
        
        eeg=zeros(length(chselect),numel);
        numel=0;
        numelm=0;
        while ~feof(datafile),
            [data,count] = fread(datafile,[numchannel,buffersize],'int16');
            if count~=0
                numelm = count/numchannel;
                eeg(:,numel+1:numel+numelm) = data(chselect,:);
                numel = numel+numelm;
            end
        end
        fclose(datafile);
        end

        if nargin == 4

        % the real buffer will be buffersize * numch * 2 bytes
        % (short = 2bytes)
        
        buffersize = 4096;
        
        % get file size, and calculate the number of samples per channel
        fileinfo = dir(fname);
        numel = ceil(fileinfo(1).bytes / 2 / numchannel);
        
        datafile = fopen(fname,'r');
        
        mmm = sprintf('%d elements',numel);
        %  disp(mmm);  
        
        eeg=zeros(length(chselect),numel);
        numel=0;
        numelm=0;
        while ~feof(datafile),
            [data,count] = fread(datafile,[numchannel,buffersize],'int16');
            if count~=0
                numelm = count/numchannel;
                eeg(:,numel+1:numel+numelm) = data(chselect,:)-repmat(data(subtract_channel,:),length(chselect),1);
                numel = numel+numelm;
            end
        end
        fclose(datafile);
        end


        eeg = eeg';
        """
        assert chselect is None, "Not all functionality from the MATLAB version is implemented!"
        assert len(args) == 0, "Not all functionality from the MATLAB version is implemented!"
        with open(fname, 'rb') as fid:
            loaded_eeg_data = np.fromfile(fid, np.int16).reshape((-1, numchannel)) #.T
        return loaded_eeg_data

    @staticmethod
    def _downsample_data(data, fs, downsampled_fs):
        # Dowsampling
        if fs > downsampled_fs:
            downsampled_pts = np.linspace(0, data.shape[0]-1, int(np.round(data.shape[0]/fs*downsampled_fs))).astype(int)
            downsampled_data = data[downsampled_pts, :]

        # Upsampling
        elif fs < downsampled_fs:
            print("Original sampling rate below 1250 Hz!")
            return None
        else:
            print("Original sampling rate equals 1250 Hz!")
            downsampled_data = data

        # Change from int16 to float16 if necessary
        # int16 ranges from -32,768 to 32,767
        # float16 has ±65,504, with precision up to 0.0000000596046
        if downsampled_data.dtype != 'float16':
            downsampled_data = np.array(downsampled_data, dtype="float16")

        return downsampled_data

    @staticmethod
    def _z_score_normalization(data):
        channels = range(np.shape(data)[1])
        for channel in channels:
            # Since data is in float16 type, we make it smaller to avoid overflows
            # and then we restore it.
            # Mean and std use float64 to have enough space
            # Then we convert the data back to float16
            dmax = np.amax(data[:, channel])
            dmin = abs(np.amin(data[:, channel]))
            dabs = dmax if dmax>dmin else dmin
            m = np.mean(data[:, channel] / dmax, dtype='float64') * dmax
            s = np.std(data[:, channel] / dmax, dtype='float64') * dmax
            s = 1 if s == 0 else s # If std == 0, change it to 1, so data-mean = 0
            data[:, channel] = ((data[:, channel] - m) / s).astype('float16')

        return data


    ## Batch
    # Do Once:
    @classmethod
    def load_eeg_data(cls, active_session_folder=Path('/content/drive/Shareddrives/Diba Lab Data/KDIBA/gor01/one/2006-6-08_14-26-15'), numchannel:int=96):
        # active_session_folder = Path('/content/drive/Shareddrives/Diba Lab Data/KDIBA/gor01/one/2006-6-08_14-26-15')
        active_session_stem = active_session_folder.stem # '2006-6-08_14-26-15'
        active_session_eeg_data_filepath = active_session_folder.joinpath(active_session_stem).with_suffix('.eeg')
        print(f'active_session_folder: {active_session_folder}')
        print(f'active_session_eeg_data_filepath: {active_session_eeg_data_filepath}')
        assert active_session_eeg_data_filepath.exists() and active_session_eeg_data_filepath.is_file()
        # nChannels = session.extracellular.nChannels
        # srLfp = session.extracellular.srLfp  
        # numchannel = 96
        # srLfp = 1250
        loaded_eeg_data = cls.readmulti(active_session_eeg_data_filepath, numchannel)
        return loaded_eeg_data, active_session_eeg_data_filepath, active_session_folder

    @classmethod
    def compute_ripples(cls, model, loaded_eeg_data, active_shank_channels_lists, 
        srLfp = 1250, downsampled_fs = 1250, 
        overlapping = True, window_size = 0.0128, stride = 0.0064, # window parameters
        ripple_detection_threshold=0.7, out_all_ripple_results=None):
        
        ## Create Empty Output:
        computation_params = None
        # out_all_ripple_results = None


        def _run_batch_cycle(shank, active_shank_channels, srLfp, loaded_eeg_data):
            """ captures:
            srLfp, downsampled_fs
            overlapping, window_size, stride
            """
            ## Begin:
            if isinstance(active_shank_channels, list):
                active_shank_channels = np.array(active_shank_channels) # convert to a numpy array
            
            # Subtract 1 from each element to get a channel index
            active_shank_channels = active_shank_channels - 1

            fs = srLfp
            loaded_data = loaded_eeg_data[:,active_shank_channels]

            print("Shape of loaded data: ", np.shape(loaded_data))
            # Downsample data
            
            print("Downsampling data from %d Hz to %d Hz..."%(fs, downsampled_fs), end=" ")
            data = cls._downsample_data(loaded_data, fs, downsampled_fs)
            print("Done!")

            # Normalize it with z-score
            print("Normalizing data...", end=" ")
            data = cls._z_score_normalization(data)
            print("Done!")

            print("Shape of loaded data after downsampling and z-score: ", np.shape(data))
            
            print("Generating windows...", end=" ")
            if overlapping:
                # Separate the data into 12.8ms windows with 6.4ms overlapping
                X = generate_overlapping_windows(data, window_size, window_stride, downsampled_fs)
            else:
                window_stride = window_size
                X = np.expand_dims(data, 0)
            print("Done!")

            print("Detecting ripples...", end=" ")
            predictions = model.predict(X, verbose=True)
            print("Done!")

            # This threshold can be changed
            
            print("Getting detected ripples indexes and times...", end=" ")
            pred_indexes = get_predictions_indexes(data, predictions, window_size=window_size, stride=window_stride, fs=downsampled_fs, threshold=ripple_detection_threshold)
            pred_times = pred_indexes / downsampled_fs
            print("Done!")

            ## Single result output
            return {'shank':shank, 'channels': active_shank_channels, 'predictions': predictions, 'pred_indexes': pred_indexes, 'pred_times': pred_times}
            
        ## Create new global result containers IFF they don't already exists
        if computation_params is None:
            computation_params = dict(overlapping = True, window_size = 0.0128, stride=0.0064, threshold=ripple_detection_threshold, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        if out_all_ripple_results is None:
            out_all_ripple_results = {'computation_params': computation_params, 'results': dict()}

        # shank = 0
        # active_shank_channels = [72,73,74,75,76,77,78,79]
        # shank = 1
        # active_shank_channels = [81,82,83,84,85,86,87,88]
        # shank = 2
        # active_shank_channels = [89,90,91,92,93,94,95,96]
        # shank = 3
        # active_shank_channels = [57,58,59,60,61,62,63,64]
        # ...
        

        for shank, active_shank_channels in enumerate(active_shank_channels_lists):
            print(f'working on shank {shank} with channels: {active_shank_channels}...')
            try:
                out_result = _run_batch_cycle(shank, active_shank_channels, loaded_eeg_data=loaded_eeg_data, srLfp=srLfp)
                out_all_ripple_results['results'][shank] = out_result
            except ValueError as e:
                out_result = {} # empty output result
                print(f'skipping shank {shank} with too many values ({len(active_shank_channels)}, expecting exactly 8).') 
                # raise e

        print(f'done with all!')
        # out_all_ripple_results
        return out_all_ripple_results

## Start Qt event loop
if __name__ == '__main__':
    # model_path = r'C:\Users\pho\repos\cnn-ripple\model'
    # g_drive_session_path = Path('/content/drive/Shareddrives/Diba Lab Data/KDIBA/gor01/one/2006-6-08_14-26-15')
    # active_local_session_path = Path(r'W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15')
    # # active_shank_channels_lists = [[72,73,74,75,76,77,78,79], [81,82,83,84,85,86,87,88], [89,90,91,92,93,94,95,96], [57,58,59,60,61,62,63,64], [41,42,43,44,45,46,47,48], [33,34,35,36,37,38,39,40], [33,34,35,36,37,38,39,40], [9,10,11,12,13,14,15,16],
    # #              [49, 50, 51, 52, 53, 54, 55, 56, 25, 26, 27, 28, 29, 30, 31, 32, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24], 
    # #             [65,66,67,68,69,70,71,72]]

    # active_shank_channels_lists = [[72,73,74,75,76,77,78,79], [81,82,83,84,85,86,87,88], [89,90,91,92,93,94,95,96], [57,58,59,60,61,62,63,64], [41,42,43,44,45,46,47,48], [33,34,35,36,37,38,39,40], [33,34,35,36,37,38,39,40],
    #             [9,10,11,12,13,14,15,16],
    #             [49, 50, 51, 52, 53, 54, 55, 56], 
    #             [25, 26, 27, 28, 29, 30, 31, 32], 
    #             [1, 2, 3, 4, 5, 6, 7, 8],
    #             [17, 18, 19, 20, 21, 22, 23, 24], 
    #             [65,66,67,68,69,70,71,72]]

    # _test_active_shank_channels_lists = np.array(active_shank_channels_lists).flatten()
    # print(f'_test_active_shank_channels_lists: {_test_active_shank_channels_lists}\n {np.shape(_test_active_shank_channels_lists)}') # (104,)
    
    

    
    # numchannel=104

    active_local_session_path = Path(r'W:\Data\KDIBA\gor01\one\2006-6-13_14-42-6')
    numchannel=96
    active_shank_channels_lists = [[2, 3, 4, 5, 6],
         [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
         [19, 20, 21, 22], 
        # [23, 24, 25], 
        [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        #  [40],
         [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
         [56, 57, 58, 59, 60]]

    active_shank_channels_lists = [a_list[:8] for a_list in active_shank_channels_lists if len(a_list)>=8]
    
    print(f'active_shank_channels_lists: {active_shank_channels_lists}')
    test_detector = ExtendedRippleDetection(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    ripple_df, out_all_ripple_results = test_detector.compute(active_session_folder=active_local_session_path, numchannel=numchannel, srLfp=1250, 
            active_shank_channels_lists=active_shank_channels_lists, overlapping=False)

    # out_all_ripple_results
    ripple_df.to_pickle(active_local_session_path.joinpath('ripple_df.pkl'))
    print(f'done. Exiting.')