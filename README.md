Traffic_Generation Project Documentation
0. Raw Data Source
Source: The raw traffic data is based on the 5GT-gen dataset (GitHub link), which provides per-second 5G traffic traces collected from real networks.

Time Resolution: 1 timestep = 1 second.

Recorded Variables:

timestamp (seconds)

Packet header information

Source/destination addresses

DL_bitrate (bps): downlink bitrate per BS per timestep

Applications Measured: Video streaming (Netflix, YouTube, Amazon Prime Video), live streaming (Naver NOW, AfreecaTV), video conferencing (Zoom, MS Teams, Google Meet), metaverse (Zepeto, Roblox), online and cloud gaming.

Dataset Size: ~328 hours (~1.18M timesteps), file sizes 0.1 GB – 7 GB.

1. Environment Setup
Project Root: Detected automatically via os.path.dirname(os.path.abspath(__file__)).

Directories:

raw_data/ → contains the original CSV files (*_dataset.csv).

generated_traffic/ → output folder for all generated CSV files.

Random Seed: Fixed (np.random.seed(42)) to ensure reproducibility.

Output: traffic.csv, bs_info.csv, user_info.csv, episodes.csv, rate_requirements.csv, uav_home_positions.csv, config.yaml.

2. Raw Data Processing
Retrieve Raw Data Files:

Function: get_all_raw_data_files()

Scans raw_data/ for all CSV files ending with _dataset.csv.

Load and Aggregate DL Bitrate:

Function: load_total_DL()

Sums all DL_bitrate columns per BS and then across all files to get total network traffic per timestep.

Output: 1D array total_dl[timestep] in bps.

3. Traffic Generation
Function: generate_traffic(raw_data_dir, output_dir, grid_rows, grid_cols)

Goal: Generate per-BS traffic time series for the entire grid, incorporating:

Dynamic hotspots/coldspots

Daily traffic patterns (rush hours, nighttime, normal)

Random event spikes

Method:

Initialize base “hotness” per BS.

Apply softmax normalization to distribute total_dl[t] among BSs.

Multiply by daily pattern, event spike, and small Gaussian noise.

Output: traffic.csv with columns: timestep, BS_0_DL, BS_1_DL, …

4. Base Station (BS) Generation
Function: generate_bs(output_dir, grid_rows, grid_cols, cell_radius)

Goal: Generate BS coordinates on a grid_rows × grid_cols layout.

Output: bs_info.csv with fields: bs_id, x_pos, y_pos, cell_radius, row, col.

5. User Generation
Function: generate_users(output_dir, grid_rows, grid_cols, cell_radius, total_users)

Goal: Randomly place total_users in the grid and assign each to the nearest BS.

User Types: voice (8%), data (21%), video (71%)

Output: user_info.csv with fields: user_id, x_pos, y_pos, home_cell, user_type.

6. Episode Generation
Function: generate_episodes(output_dir, grid_rows, grid_cols, traffic_file, max_timesteps_per_episode, max_episodes)

Goal: Split total traffic into multiple episodes, adding variability:

Noise per timestep

Random hotspot/coldspot changes

Abnormal spikes

Output: episodes.csv (per-episode traffic per BS) + internal bs_status array (1=active, 0=low load).

7. Rate Requirements Generation
Function: generate_rate_requirements(output_dir, traffic_df)

Goal: Convert BS traffic into per-user rate requirements for each timestep in each episode.

Method:

Identify user’s home BS

Base rate = BS traffic / number of users attached

Multiply by service type factor: voice=0.2, data=1.0, video=3.0

Convert to Mbps, minimum 0.1 Mbps

Output: rate_requirements.csv with (episode_id, timestep, user_id, rate_requirement)

8. UAV Home Position Generation
Function: generate_uav_home_positions(output_dir, grid_rows, grid_cols, total_episodes, total_uavs, cell_radius)

Goal: Generate initial and final positions for UAVs per episode.

Note: Currently UAVs remain stationary.

Output: uav_home_positions.csv

9. Config Save
Function: save_config(output_dir, config_dict)

Saves all simulation parameters (grid size, number of users/UAVs, episode length, etc.) to config.yaml.

10. Full Pipeline Execution
Main script (if __name__ == "__main__":) executes:

Generate traffic → traffic.csv

Generate BS info → bs_info.csv

Generate user info → user_info.csv

Generate episodes → episodes.csv

Generate rate requirements → rate_requirements.csv

Generate UAV positions → uav_home_positions.csv

Save configuration → config.yaml

Output data is ready for RL simulation (UAV-BS-user control, MADDPG or similar).
