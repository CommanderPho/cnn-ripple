import os
import numpy as np
import pandas as pd
from pathlib import Path
import dill as pickle
import itertools # for list unpacking
from neuropy.utils.load_exported import LoadXml, find_session_xml # for compute_with_params_loaded_from_xml
from neuropy.utils.dynamic_container import DynamicContainer, override_dict, overriding_dict_with, get_dict_subset

from cnn_ripple.load_data import generate_overlapping_windows
from cnn_ripple.format_predictions import get_predictions_indexes
import tensorflow.keras.backend as K
import tensorflow.keras as kr



## Define the .ui file path
_path = os.path.dirname(os.path.abspath(__file__))
_modelDirectory = os.path.join(_path, '../../model')
_rebuild_ml_model_on_load = False # If True, the machine learning model is rebuilt on load for future computations. Otherwise it's just left as None


# Works around objects that were pickled with old object names producing error: # ModuleNotFoundError: No module named 'src.cnn'
class RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print(f'RenamingUnpickler.find_class(module: {module}, name: {name})')
        # if module == 'src.cnn':
        #     module = 'cnn_ripple'
        module = module.replace('src.cnn', 'cnn_ripple', 1)
        return super().find_class(module, name)


## Save result if wanted:
class ExtendedRippleDetection(object):
    """ Uses the tensorflow model to detect ripples in a given recording sessions LFPs

    Usage:
        from cnn_ripple.PhoRippleDetectionTesting import ExtendedRippleDetection, main_compute_with_params_loaded_from_xml
        from cnn_ripple.PhoRippleDetectionTesting import ExtendedRippleDetection, main_compute_with_params_loaded_from_xml

    """
    def __init__(self, **kwargs): # learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False
        super(ExtendedRippleDetection, self).__init__()
        self.active_session_folder = None
        self.active_session_eeg_data_filepath = None
        self.loaded_eeg_data = None
        self.out_all_ripple_results = None
        self._detected_ripple_epochs_df = None
        self._continuous_ripple_likelihoods_df, self._continuous_ripple_prediction_timesteps, self._continuous_ripple_shanks_prediction_values_array = None, None, None
        self.optimizer, self.model = self._load_model(**(dict(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)|kwargs))

    def compute(self, active_session_folder=Path('/content/drive/Shareddrives/Diba Lab Data/KDIBA/gor01/one/2006-6-08_14-26-15'), numchannel = 96,
            srLfp = 1250, downsampled_fs = 1250, 
            overlapping = True, window_size = 0.0128, window_stride = 0.0064, # window parameters
            ripple_detection_threshold=0.7,
            active_shank_channels_lists = [[72,73,74,75,76,77,78,79], [81,82,83,84,85,86,87,88]],
            **kwargs
            ):
        """ 

        Updates:
            active_session_folder
            loaded_eeg_data, 
            active_session_eeg_data_filepath,
            active_session_folder
            out_all_ripple_results
            detected_ripples_df
            continuous_ripple_likelihoods_df, continuous_ripple_prediction_timesteps, continuous_ripple_shanks_prediction_values_array
        """
        self.active_session_folder = active_session_folder
        self.loaded_eeg_data, self.active_session_eeg_data_filepath, self.active_session_folder = self.load_eeg_data(active_session_folder=active_session_folder, numchannel=numchannel)
        self.out_all_ripple_results = self.compute_ripples(self.model, self.loaded_eeg_data, srLfp=srLfp, downsampled_fs=downsampled_fs, overlapping=overlapping, window_size=window_size, window_stride=window_stride, ripple_detection_threshold=ripple_detection_threshold, active_shank_channels_lists=active_shank_channels_lists, out_all_ripple_results=None, **(dict(debug_trace_computations_output=True, debug_print=False)|kwargs))
        self._continuous_ripple_likelihoods_df, (self._continuous_ripple_prediction_timesteps, self._continuous_ripple_shanks_prediction_values_array) = self.build_cnn_computed_ripple_prediction_probabilities()
        
        # Save intermediate:
        # out_all_ripple_results_filepath = active_session_folder.joinpath('out_all_ripple_results.pkl')
        # with open(out_all_ripple_results_filepath, 'wb') as f:
        #     print(f'saving results to {str(out_all_ripple_results_filepath)}...')
        #     pickle.dump(self.out_all_ripple_results, f)
        # print(f'done.')
        flattened_pred_ripple_start_stop_times = np.vstack([a_result['pred_times'] for a_result in self.out_all_ripple_results['results'].values() if np.size(a_result['pred_times'])>0])
        # print(f'flattened_pred_ripple_start_stop_times: {np.shape(flattened_pred_ripple_start_stop_times)}') # (6498, 2)
        detected_ripple_epochs_df = pd.DataFrame({'start':flattened_pred_ripple_start_stop_times[:,0], 'stop': flattened_pred_ripple_start_stop_times[:,1]})
        detected_ripple_epochs_df = ExtendedRippleDetection._build_post_load_ripple_df(self.good_results.copy(), debug_print=False)
        self._detected_ripple_epochs_df = detected_ripple_epochs_df
        print(f'Saving ripple_df to csv: {self.predicted_ripples_dataframe_csv_save_filepath}')
        detected_ripple_epochs_df.to_csv(self.predicted_ripples_dataframe_csv_save_filepath)
        return detected_ripple_epochs_df, self.out_all_ripple_results

    # load/save paths ____________________________________________________________________________________________________ #
    @property
    def predicted_ripples_dataframe_csv_save_filepath(self):
        return self.active_session_folder.joinpath('pred_ripples.csv')

    @property
    def object_save_filepath(self):
        return self.active_session_folder.joinpath('ripple_detector.pkl')

    @property
    def ripple_dataframe_pickle_save_filepath(self):
        return self.active_session_folder.joinpath('ripple_df.pkl')

    # computation_params _________________________________________________________________________________________________ #
    @property
    def computation_params(self):
        return self.out_all_ripple_results.get('computation_params', None)

    # preprocessed_data __________________________________________________________________________________________________ #
    @property
    def _preprocessed_data(self):
        return self.out_all_ripple_results.get('preprocessed_data', None)
    @property
    def flattened_channels_list(self):
        return self._preprocessed_data.get('flattened_channels_list', None)
    @property
    def preprocessed_data(self):
        return self._preprocessed_data.get('data', None)

    # results ____________________________________________________________________________________________________________ #
    @property
    def results(self):
        return self.out_all_ripple_results.get('results', None)
    @property
    def shank_ids(self):
        """The good_shank_ids property."""
        return list(self.results.keys())

    @property
    def detected_ripple_epochs_df(self):
        """The detected_ripples_df property."""
        return self._detected_ripple_epochs_df
    @detected_ripple_epochs_df.setter
    def detected_ripple_epochs_df(self, value):
        self._detected_ripple_epochs_df = value


    @property
    def good_results(self):
        """The good_results property."""
        if self.out_all_ripple_results is None:
            return {}
        else:
            return {k:v for k, v in self.out_all_ripple_results['results'].items() if np.size(v['pred_times'])>0} # Exclude empty items from the output dictionary 
    @property
    def good_shank_ids(self):
        """The good_shank_ids property."""
        return list(self.good_results.keys())


    # Continuous Ripple Likelihood Properties ____________________________________________________________________________ #



    @property
    def has_continuous_computation_results(self):
        """ """
        try:
            if self._continuous_ripple_likelihoods_df is None:
                return False
            else:
                return True
        except AttributeError as e:
            return False


    @property
    def continuous_ripple_likelihoods_df(self):
        """ """
        if not self.has_continuous_computation_results:
            self._continuous_ripple_likelihoods_df, (self._continuous_ripple_prediction_timesteps, self._continuous_ripple_shanks_prediction_values_array) = self.build_cnn_computed_ripple_prediction_probabilities()
        return self._continuous_ripple_likelihoods_df

    @property
    def continuous_ripple_prediction_timesteps(self):
        """ """
        if not self.has_continuous_computation_results:
            self._continuous_ripple_likelihoods_df, (self._continuous_ripple_prediction_timesteps, self._continuous_ripple_shanks_prediction_values_array) = self.build_cnn_computed_ripple_prediction_probabilities()
        return self._continuous_ripple_prediction_timesteps

    @property
    def continuous_ripple_shanks_prediction_values_array(self):
        """ """
        if not self.has_continuous_computation_results:
            self._continuous_ripple_likelihoods_df, (self._continuous_ripple_prediction_timesteps, self._continuous_ripple_shanks_prediction_values_array) = self.build_cnn_computed_ripple_prediction_probabilities()
        return self._continuous_ripple_shanks_prediction_values_array



    


    # ==================================================================================================================== #
    # Helpers                                                                                                              #
    # ==================================================================================================================== #
    def _load_model(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False):
        print("Loading CNN model...", end=" ")
        optimizer = kr.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)
        # model_path = "../../model"
        # model_path = r"C:\Users\pho\repos\cnn-ripple\model"
        model_path = _modelDirectory
        model = kr.models.load_model(model_path, compile=False)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
        print("Done!")
        return optimizer, model


    ## Continuous prediction probabilities:
    def build_cnn_computed_ripple_prediction_probabilities(self, shank_ids=None, debug_print=False):  
        return ExtendedRippleDetection._build_cnn_computed_ripple_prediction_probabilities(self.out_all_ripple_results.copy(), shank_ids=shank_ids, debug_print=debug_print)


    @classmethod
    def _build_cnn_computed_ripple_prediction_probabilities(cls, out_all_ripple_results, shank_ids=None, debug_print=False):  
        """ Builds the timestamps that correspond with each probability prediction and then builds a dataframe containing the probability predictions for each shank

        Returns:
            pd.DataFrame 329398 rows × 13 columns
            (prediction_timesteps, shanks_prediction_values_array): 
                prediction_timesteps: a (num_timestamps,) np.array - e.g. (329398,)
                shanks_prediction_values_array is an (num_timestamps, numShanks) np.array - e.g. (329398, 12)
        Usage:
            ripple_predictions_df, (prediction_timesteps, shanks_prediction_values_array) = loaded_ripple_detector.build_cnn_computed_ripple_prediction_probabilities()
            ripple_predictions_df

        """
        if not isinstance(out_all_ripple_results, DynamicContainer):
            out_all_ripple_results = DynamicContainer.init_from_dict(out_all_ripple_results)

        valid_shank_ids = list(out_all_ripple_results.results.keys())

        if shank_ids is None:
            # all shank ids by default:
            shank_ids = valid_shank_ids.copy()

        dt = out_all_ripple_results.computation_params['stride'] # same for all shank_ids
        prediction_timesteps = None
        prediction_values_dict = {}

        for a_shank_id in shank_ids:
            # list(out_all_ripple_results.keys()) # ['computation_params', 'results']
            assert a_shank_id in out_all_ripple_results.results, f"{a_shank_id} is not in results list: {valid_shank_ids}"
            a_result = out_all_ripple_results.results[a_shank_id]
            # a_result['predictions'].shape # (329925, 1, 1)
            # list(out_all_ripple_results.results.keys()) # [0, 1, 2, 3]

            prediction_values = np.squeeze(a_result['predictions'])
            num_prediction_timestamps = np.size(prediction_values)
            if prediction_timesteps is None:
                prediction_timesteps = np.arange(num_prediction_timestamps) * dt # + curr_active_pipeline.sess.t_start
            else:
                assert np.shape(prediction_timesteps)[0] == num_prediction_timestamps, "each shank should have the same number of timestamps, right??"


            prediction_values_dict[f'v{a_shank_id}'] = prediction_values
        if debug_print:
            print(f'prediction_values_dict: {prediction_values_dict}')
    
        ripple_predictions_df = pd.DataFrame({'t': prediction_timesteps, **prediction_values_dict})
        # concatenate each of the values columns into a (num_timestamps, numShanks) output array
        shanks_prediction_values_array = np.stack(list(prediction_values_dict.values()), axis=1) #.shape # (329398, 12)
        return ripple_predictions_df, (prediction_timesteps, shanks_prediction_values_array)


    # def build_post_load_ripple_df(self, debug_print=False):
    #     return ExtendedRippleDetection._build_post_load_ripple_df(self.good_results.copy(), debug_print=debug_print)

    @classmethod
    def _build_post_load_ripple_df(cls, out_all_ripple_results_good_results, debug_print=False):
        """ adds the 'shank_idx'
            out_all_ripple_results = loaded_ripple_detector.out_all_ripple_results.copy()
            ripple_df = _build_post_load_ripple_df(out_all_ripple_results)
        """
        flattened_pred_ripple_shank_idxs = np.hstack([np.full_like(np.squeeze(a_result['pred_times'][:,0]), a_result['shank'], dtype=np.int16) for a_result in out_all_ripple_results_good_results.values()])
        if debug_print:
            print(np.shape(flattened_pred_ripple_shank_idxs)) # (6016,)
        flattened_pred_ripple_start_stop_times = np.vstack([a_result['pred_times'] for a_result in out_all_ripple_results_good_results.values() if np.size(a_result['pred_times'])>0])
        if debug_print:    
            print(f'flattened_pred_ripple_start_stop_times: {np.shape(flattened_pred_ripple_start_stop_times)}') # (6498, 2)
        ripple_df = pd.DataFrame({'start':flattened_pred_ripple_start_stop_times[:,0], 'stop': flattened_pred_ripple_start_stop_times[:,1], 'shank_idx': flattened_pred_ripple_shank_idxs})
        ripple_df = ripple_df.sort_values(by=['start']) # sort the values by the start time
        return ripple_df

    # ==================================================================================================================== #
    # Persistance Saving/Loading                                                                                           #
    # ==================================================================================================================== #
    def __getstate__(self):
        """Used for serializing instances"""  
        # start with a copy so we don't accidentally modify the object state
        # or cause other conflicts
        state = self.__dict__.copy()

        # remove unpicklable entries
        # test_detector.model # keras.engine.sequential.Sequential
        # test_detector.optimizer # keras.optimizers.optimizer_v2.adam.Adam
        del state['model']
        del state['optimizer']

        return state

    def __setstate__(self, state):
        """Used for deserializing"""
        # restore the state which was picklable
        self.__dict__.update(state)
        
        ## restore unpicklable entries (self.optimizer, self.model)
        if self.out_all_ripple_results is not None:
            loaded_model_kwargs = get_dict_subset(self.out_all_ripple_results['computation_params'], ['learning_rate', 'beta_1', 'beta_2', 'epsilon', 'amsgrad'])
        else:
            loaded_model_kwargs = {} # empty dict to use defaults
        # Rebuild the model:
        if _rebuild_ml_model_on_load:
            self.optimizer, self.model = self._load_model(**(dict(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)|loaded_model_kwargs))
        else:
            print(f'WARNING: _rebuild_ml_model_on_load is set to False, so not rebuilding the machine learning model after loading.')
            self.optimizer, self.model = None, None


    def save(self):
        """pickles the whole ExtendedRippleDetection object to file"""
        out_ripple_detector_filepath = self.object_save_filepath
        with open(out_ripple_detector_filepath, 'wb') as f:
            print(f'saving entire ripple detector object to {str(out_ripple_detector_filepath)}...')
            pickle.dump(self, f)
        print(f'done.')

    @classmethod
    def load(cls, in_ripple_detector_filepath):
        """Unpickle the object from file
        Usage:
            from cnn_ripple.PhoRippleDetectionTesting import ExtendedRippleDetection, main_compute_with_params_loaded_from_xml
            in_ripple_detector_filepath = Path(r'W:\Data\KDIBA\gor01\one\2006-6-07_11-26-53\ripple_detector.pkl')
            loaded_ripple_detector = ExtendedRippleDetection.load(in_ripple_detector_filepath)
            loaded_ripple_detector

        """
        if not isinstance(in_ripple_detector_filepath, Path):
            in_ripple_detector_filepath = Path(in_ripple_detector_filepath)

        # If the user passes in a directory instead of the direct file path to the pickle file, find the pickle file using the default name
        if in_ripple_detector_filepath.is_dir():
            # assume the passed in directory is the active_local_session_path
            print(f'assuming the passed in directory is the active_local_session_path, building default in_ripple_detector_filepath from it...')
            active_local_session_path = in_ripple_detector_filepath
            in_ripple_detector_filepath = active_local_session_path.joinpath('ripple_detector.pkl') # Path(r'W:\Data\KDIBA\gor01\one\2006-6-07_11-26-53\ripple_detector.pkl')
            print(f'\t in_ripple_detector_filepath: {str(in_ripple_detector_filepath)}')

        with open(in_ripple_detector_filepath, 'rb') as f:
            print(f'loading pickled ripple detector object from {str(in_ripple_detector_filepath)}...')
            try:
                loaded_ripple_detector = pickle.load(f)
            except ModuleNotFoundError as e:
                print(f'encountered old class ({e}). Trying RenamingUnpickler...')
                loaded_ripple_detector = RenamingUnpickler(f).load()

        print(f'done.')
        return loaded_ripple_detector

    # ==================================================================================================================== #
    # Class and Static Methods                                                                                             #
    # ==================================================================================================================== #

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
            print("Downsampling data from %d Hz to %d Hz..."%(fs, downsampled_fs), end=" ")
            downsampled_pts = np.linspace(0, data.shape[0]-1, int(np.round(data.shape[0]/fs*downsampled_fs))).astype(int)
            downsampled_data = data[downsampled_pts, :]

        # Upsampling
        elif fs < downsampled_fs:
            print("Original sampling rate below 1250 Hz!")
            return None
        else:
            # print("Original sampling rate equals 1250 Hz!")
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

    @classmethod
    def _run_single_shank_computation(cls, model, loaded_eeg_data, active_shank, active_shank_channels, srLfp, downsampled_fs, overlapping, window_size, window_stride, ripple_detection_threshold, debug_trace_computations_output=False, debug_print=False):
        """ Runs a single set of 8 channels (from one 8-channel probe)
        """
        ## Begin:
        if isinstance(active_shank_channels, list):
            active_shank_channels = np.array(active_shank_channels) # convert to a numpy array
        
        # Subtract 1 from each element to get a channel index
        active_shank_channels = active_shank_channels - 1

        fs = srLfp

        # Get the subset of the data corresponding to only the active channels 
        loaded_data = loaded_eeg_data[:,active_shank_channels]
        if debug_print:
            print("Shape of loaded data: ", np.shape(loaded_data))
        # Downsample data (if needed)
        data = cls._downsample_data(loaded_data, fs, downsampled_fs)
        if debug_print:
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

        print("Getting detected ripples indexes and times...", end=" ")
        pred_indexes = get_predictions_indexes(data, predictions, window_size=window_size, stride=window_stride, fs=downsampled_fs, threshold=ripple_detection_threshold)
        pred_times = pred_indexes / downsampled_fs
        print("Done!")

        ## Single result output
        curr_shank_computations = {'shank':active_shank, 'channels': active_shank_channels, 'time_windows': X, 'predictions': predictions, 'pred_indexes': pred_indexes, 'pred_times': pred_times}
        if debug_trace_computations_output:
            curr_shank_computations['data'] = data

        return curr_shank_computations


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
        overlapping = True, window_size = 0.0128, window_stride = 0.0064, # window parameters
        ripple_detection_threshold=0.7, debug_trace_computations_output=False, out_all_ripple_results=None, debug_print=False):
        
        ## Create Empty Output:
        computation_params = None

        ## Create new global result containers IFF they don't already exists
        if computation_params is None:
            computation_params = dict(overlapping = True, window_size = 0.0128, stride=0.0064, threshold=ripple_detection_threshold, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        if out_all_ripple_results is None:
            out_all_ripple_results = {'computation_params': computation_params, 'results': dict()}


        ## Pre-process the data

        # flattened_channels_list is a flat list of channels (not partitioned into lists of 8 channels corresponding to a single probe)
        flattened_channels_list = list(itertools.chain.from_iterable(active_shank_channels_lists))

        print("Shape of loaded data: ", np.shape(loaded_eeg_data))
        # Downsample data
        downsampled_loaded_eeg_data = cls._downsample_data(loaded_eeg_data, srLfp, downsampled_fs)
        post_downsampling_srLfp = downsampled_fs # after downsampling data, the passed srLfp should be set to the downsampled rate so it isn't downsampled again

        out_all_ripple_results['preprocessed_data'] = {'data':downsampled_loaded_eeg_data, 'post_downsampling_srLfp': post_downsampling_srLfp, 'flattened_channels_list': flattened_channels_list}
        print("Done!")


        # shank = 0
        # active_shank_channels = [72,73,74,75,76,77,78,79]
        # shank = 1
        # active_shank_channels = [81,82,83,84,85,86,87,88]
        # ...
        
        for active_shank, active_shank_channels in enumerate(active_shank_channels_lists):
            print(f'working on shank {active_shank} with channels: {active_shank_channels}...')
            try:
                out_result = cls._run_single_shank_computation(model, downsampled_loaded_eeg_data, active_shank, active_shank_channels, srLfp=post_downsampling_srLfp, downsampled_fs=downsampled_fs,
                    overlapping=overlapping, window_size=window_size, window_stride=window_stride, ripple_detection_threshold=ripple_detection_threshold,
                    debug_trace_computations_output=debug_trace_computations_output, debug_print=debug_print)
                out_all_ripple_results['results'][active_shank] = out_result
            except ValueError as e:
                out_result = {} # empty output result
                print(f'skipping shank {active_shank} with too many values ({len(active_shank_channels)}, expecting exactly 8).') 

        print(f'done with all!')
        return out_all_ripple_results


# ==================================================================================================================== #
# Start MAIN                                                                                                           #
# ==================================================================================================================== #

def main_compute_with_params_loaded_from_xml(local_session_path, whitelisted_shank_ids=None, **kwargs):
    """Loads the session recording info from the XML located in the local_session_path (session folder), and the computes the ripples from that data

    Args:
        local_session_path (_type_): _description_
        whitelisted_shank_ids (list): only include the specified shank IDS, e.g. [0, 1, 2]
    Returns:
        _type_: _description_

    Usage:
        from cnn_ripple.PhoRippleDetectionTesting import ExtendedRippleDetection, main_compute_with_params_loaded_from_xml

        # local_session_path = Path(r'W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15')
        # local_session_path = Path(r'W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15')
        local_session_path = Path(r'W:\Data\KDIBA\gor01\one\2006-6-13_14-42-6')
        ripple_df, out_all_ripple_results, out_all_ripple_results = main_compute_with_params_loaded_from_xml(local_session_path)


    """
    session_xml_filepath, session_stem, local_session_path = find_session_xml(local_session_path)
    out_xml_dict, d = LoadXml(session_xml_filepath)
    active_shank_channels_lists = out_xml_dict['AnatGrps']
    if whitelisted_shank_ids is not None:
        # only include the specified shank IDS, e.g. [0, 1, 2]
        print(f'including only WHITELISTED shank IDS: {whitelisted_shank_ids}')
        active_shank_channels_lists = [active_shank_channels_lists[i] for i in whitelisted_shank_ids]
    
    print(f"active_shank_channels_lists: {active_shank_channels_lists}")

    ## Build the detector:
    active_detector = ExtendedRippleDetection(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    ripple_df, out_all_ripple_results = active_detector.compute(**({'active_session_folder': local_session_path,
         'numchannel': out_xml_dict['nChannels'], 'srLfp': out_xml_dict['lfpSampleRate'], 'active_shank_channels_lists': out_xml_dict['AnatGrps'],
         'overlapping': True, 'window_size': 0.0128, 'window_stride': 0.0064} | kwargs))

    # out_all_ripple_results
    ripple_df.to_pickle(local_session_path.joinpath('ripple_df.pkl'))
    print(f'done. Exiting.')

    # Save the main object
    active_detector.save()

    return active_detector, ripple_df, out_all_ripple_results


if __name__ == '__main__':
    # model_path = r'C:\Users\pho\repos\cnn-ripple\model'
    # g_drive_session_path = Path('/content/drive/Shareddrives/Diba Lab Data/KDIBA/gor01/one/2006-6-08_14-26-15')

    local_session_parent_path = Path(r'W:\Data\KDIBA\gor01\one')
    local_session_names_list = ['2006-6-07_11-26-53', '2006-6-08_14-26-15', '2006-6-09_1-22-43', '2006-6-09_3-23-37', '2006-6-12_15-55-31', '2006-6-13_14-42-6']
    local_session_paths_list = [local_session_parent_path.joinpath(a_name).resolve() for a_name in local_session_names_list]

    active_local_session_path: Path = local_session_paths_list[1]
    test_detector, ripple_df, out_all_ripple_results = main_compute_with_params_loaded_from_xml(active_local_session_path)


    # # active_shank_channels_lists = [a_list[:8] for a_list in active_shank_channels_lists if len(a_list)>=8]
    