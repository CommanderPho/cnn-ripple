#@markdown This is an interactive plot of the loaded data, where detected ripples are shown in blue. Data is displayed in chunks of 1 seconds and you can **move forward, backwards or jump to an specific second** using the control bar at the bottom.\
#@markdown \
#@markdown Run this cell to load the plotting method. Execute the **following** cell to use the method.
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def plot_ripples(data, pred_indexes, k, downsampled_fs=1250):
  data_size = data.shape[0]
  data_dur = data_size / downsampled_fs
  times = np.arange(data_size) / downsampled_fs

  if k >= times[-1]:
    print(f"Data is only %ds long!"%(times[-1]))
    return
  elif k < 0:
    print("Please introduce a valid integer.")
    return

  ini_idx = int(k * downsampled_fs)
  end_idx = np.minimum(int((k+1) * downsampled_fs), data_size-1)

  pos_mat = list(range(data.shape[1]-1, -1, -1)) * np.ones((end_idx-ini_idx, data.shape[1]))

  fig = plt.figure(figsize=(9.75,5))
  ax = fig.add_subplot(1, 1, 1)
  ax.set_ylim(-3, 9)
  ax.margins(x=0)
  plt.tight_layout()
  plt.xlabel("Time (s)")

  lines = ax.plot(times[ini_idx:end_idx], data[ini_idx:end_idx, :]*1/np.max(data[ini_idx:end_idx, :], axis=0) + pos_mat, color='k', linewidth=1)

  fills = []
  for pred in pred_indexes:
      if (pred[0] >= ini_idx and pred[0] <= end_idx) or (pred[1] >= ini_idx and pred[1] <= end_idx):
          rip_ini = (pred[0]) / downsampled_fs
          rip_end = (pred[1]) / downsampled_fs
          fill = ax.fill_between([rip_ini, rip_end], [-3, -3], [9, 9], color="tab:blue", alpha=0.3)
          fills.append(fill)

  plt.show()



# ==================================================================================================================== #
# Start MAIN                                                                                                           #
# ==================================================================================================================== #
if __name__ == '__main__':
    local_session_parent_path = Path(r'W:\Data\KDIBA\gor01\one')
    local_session_names_list = ['2006-6-07_11-26-53', '2006-6-08_14-26-15', '2006-6-09_1-22-43', '2006-6-09_3-23-37', '2006-6-12_15-55-31', '2006-6-13_14-42-6']
    local_session_paths_list = [local_session_parent_path.joinpath(a_name).resolve() for a_name in local_session_names_list]

    active_local_session_path: Path = local_session_paths_list[0]
    test_detector, ripple_df, out_all_ripple_results, out_all_ripple_results = main_compute_with_params_loaded_from_xml(active_local_session_path)


    # # active_shank_channels_lists = [a_list[:8] for a_list in active_shank_channels_lists if len(a_list)>=8]
    
    print("Loaded!")


    #@title Time (in seconds) { run: "auto", vertical-output: true, display-mode: "form" }
    second =  0#@param {type:"integer"}

    # plot_ripples(second)
    _out = plot_ripples(data, pred_indexes, second, downsampled_fs=1250)

