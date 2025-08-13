# Traffic_Generation Project

## Overview
This project generates per-BS traffic time series and simulation data for UAV-BS-user control experiments (e.g., MADDPG).  
It processes raw 5G traffic data and produces all necessary outputs for reinforcement learning simulations.

## 0. Raw Data Source
- **Source:** 5GT-gen dataset ([GitHub link](https://github.com/0913ktg/5G-Traffic-Generator))  
- **Time Resolution:** 1 timestep = 1 second  
- **Recorded Variables:** timestamp, packet headers, source/destination, DL_bitrate per BS  
- **Applications:** Video streaming (Netflix, YouTube, Amazon Prime Video), live streaming (Naver NOW, AfreecaTV), video conferencing (Zoom, MS Teams, Google Meet), metaverse (Zepeto, Roblox), online/cloud gaming  
- **Dataset Size:** ~328 hours (~1.18M timesteps), file sizes 0.1 GB – 7 GB

> **Note:** Large CSV files in `raw_data/` and `generated_traffic/` are tracked via Git LFS. Ensure Git LFS is installed.

## 1. Project Structure
Traffic_Generation/  
├─ generate_traffic.py # Main script  
├─ README.md # Project documentation  
├─ raw_data/ # Original CSV datasets  
├─ generated_traffic/ # Generated outputs  
├─ config.yaml # Saved simulation parameters  
└─ .gitattributes # LFS configuration  

## 2. Environment Setup
- **Python libraries:** `numpy`, `pandas`, `pyyaml`  
- **Random Seed:** Fixed (`np.random.seed(42)`) for reproducibility

**Optional setup using virtual environment:**
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install numpy pandas pyyaml
```

## 3. Full Pipeline
Running generate_traffic.py executes the following steps:
- Traffic Generation
  + Function: generate_traffic(raw_data_dir, output_dir, grid_rows, grid_cols)
  + Outputs traffic.csv with per-BS traffic time series
  + Incorporates dynamic hotspots, daily patterns, random event spikes, Gaussian noise
- Base Station (BS) Generation
  + Function: generate_bs(output_dir, grid_rows, grid_cols, cell_radius)
  + Outputs bs_info.csv with BS positions and properties
- User Generation
  + Function: generate_users(output_dir, grid_rows, grid_cols, cell_radius, total_users)
  + Outputs user_info.csv with user locations, types, and assigned BS
- Episode Generation
  + Function: generate_episodes(output_dir, grid_rows, grid_cols, traffic_file, max_timesteps_per_episode, max_episodes)
  + Outputs episodes.csv with per-episode traffic per BS
- Rate Requirements Generation
  + Function: generate_rate_requirements(output_dir, traffic_df)
  + Outputs rate_requirements.csv for per-user RL simulation
- UAV Home Position Generation
  + Function: generate_uav_home_positions(output_dir, grid_rows, grid_cols, total_episodes, total_uavs, cell_radius)
  + Outputs uav_home_positions.csv (stationary UAVs)
- Config Save
  + Function: save_config(output_dir, config_dict)
  + Saves all simulation parameters to config.yaml

## 4. How to Run
```
# Clone repository
git clone <repo_url>
cd Traffic_Generation

# Run the full pipeline
python generate_traffic.py
```

After execution, the following files will be generated in generated_traffic/:
- traffic.csv
- bs_info.csv
- user_info.csv
- episodes.csv
- rate_requirements.csv
- uav_home_positions.csv
- config.yaml
  
These outputs are ready for UAV-BS-user control simulations using MADDPG or similar RL algorithms.

## 5. Notes
- Ensure Git LFS is installed to handle large CSV files.
- All random processes use a fixed seed for reproducibility.
- The UAVs are stationary in the current setup; mobility can be added if needed.
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/95fdf78d-c759-42b7-9809-99f824f44290" />
