import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.suiteHeart_to_StrainNet import prepare_StrainNet_data_from_SuiteHeart

import os
import copy
import shutil
import scipy.io as sio
import skimage.io as skio
st.set_page_config(
    page_title="suiteHEART® Data Utility",
    page_icon="🫀",
    # layout="wide",
    # initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

st.title("🫀 suiteHEART® Data Utility")
st.markdown(
    '''
    Tools to analyze suiteHEART® data and prepare it for [EpsteinLabUVA/StrainNet](https://github.com/EpsteinLabUVA/StrainNet) training.  
    Developed by [Jiarui Xing](https://scholar.google.com/citations?user=G_0diKUAAAAJ&hl=en), PhD Student, [MIA lab](https://www.cs.virginia.edu/~mz8rr/research.html)@University of Virginia,  
    in collaboration with the [Epstein Lab](https://engineering.virginia.edu/faculty/frederick-h-epstein)@University of Virginia.
    '''
)


st.subheader('Analysis Options')
# Show analysis options
# Target size: could be integer like 128, float like 128.0, or tuple like (128, 128) or (128.0, 128.0); default is '(128,128)'
# Get raw input from user
target_size_str = st.text_input('Target size', value='(128,128)', help='The target size of the images')
# Parse the input to tuple
try:
    target_size = eval(target_size_str)
    if type(target_size) not in [int, float, tuple, list]:
        st.write(f"Please input a valid target size")
        target_size = None
    elif type(target_size) in [int, float]:
        target_size = (target_size, target_size)
    else:
        pass
except:
    st.write(f"Please input a valid target size")
    target_size = None

# Show target_pixel_resolution option
# 'ori' or '1x1' or arbitary float number or integer number; default is '1x1'
# Here we use radio button to limit the options - 'ori', '1x1', or 'custom', and 'custom' will show a text input for user to input the custom value
target_pixel_resolution_raw = st.radio('Target pixel resolution', ['ori', '1x1', 'custom'], index=1, help='The target pixel resolution of the images')
if target_pixel_resolution_raw == 'custom':
    custom_target_pixel_resolution = st.text_input('Custom target pixel resolution', value='1.0', help='The custom target pixel resolution of the images')
    try:
        target_pixel_resolution = float(custom_target_pixel_resolution)
    except:
        st.write(f"Please input a valid custom target pixel resolution")
        target_pixel_resolution = None
else:
    target_pixel_resolution = target_pixel_resolution_raw

# Visualize Options
st.subheader('Visualize Options')
# Radio: show static frame or dynamic gif visualization
dynamic_option = st.radio('Visualization', ['Static', 'Dynamic'], index=0, help='Choose the visualization mode')

# Upload a file to ./analyzed_files/uploads/
# Uploaded file
uploaded_file = st.file_uploader(
    "Choose a SuiteHeart exported `.mat` file",
    type=['mat'],
    accept_multiple_files=False,
    key='uploaded_file',
    help='Please upload a SuiteHeart exported `.mat` file')
# st.write(f"Uploaded file: {uploaded_file}")
# Move the uploaded file to ./analyzed_files/uploads/
upload_rootdir = Path("./analyzed_files/uploads")
upload_rootdir.mkdir(parents=True, exist_ok=True)

# if uploaded_file is not None:
#     # Move file
#     uploaded_filename_new = upload_rootdir / f"{uploaded_file.name}"
#     st.write(f"Move file from {uploaded_file.name} to {uploaded_filename_new}")
#     shutil.move(uploaded_file.name, uploaded_filename_new)
    # uploaded_file = uploaded_filename_new   

# Add a button to start analyze the file; this button is always there, but will be disabled until a file is uploaded
analysis_button = st.button('Analyze', key='analyze_button', disabled=uploaded_file is None, help='Please upload a file first')

# Add a button to download the analysis result; this button is always there, but will be disabled until the analysis is done
# finished_analysis = False
# https://discuss.streamlit.io/t/streamlit-button-disable-enable/31293/2
# def finished_analysis(b):
#     st.session_state["finished_analysis"] = b
# download_button = st.button('Download analysis result', key='download_button', disabled=~st.session_state.get("finished_analysis", False), help='Press the button to download the analysis result')
# https://ploomber.io/blog/streamlit_exe/
# https://wqmoran.com/streamlit-axioserror-status-code-403-solution/

# Show button to clear all files under ./analyzed_files/
slice_rootdir = Path("./analyzed_files/raw")
zip_rootdir = Path(f"./analyzed_files/zip")
vis_rootdir = Path(f"./analyzed_files/vis")
for directory in [slice_rootdir, zip_rootdir, vis_rootdir]:
    directory.mkdir(parents=True, exist_ok=True)

clear_button = st.button('Clear all files', key='clear_button', help='Press the button to clear all analyzed files')
if clear_button:
    # Remove all files under ./analyzed_files/
    shutil.rmtree(upload_rootdir)
    shutil.rmtree(slice_rootdir)
    shutil.rmtree(zip_rootdir)
    shutil.rmtree(vis_rootdir)

    # Remove all *.mat files
    for file in Path(".").glob("*.mat"):
        file.unlink()
        # os.remove(file)
        st.write(f"File {file} removed")
    st.write(f"Files cleared")

if uploaded_file is not None:
    # Make a local copy of the uploaded file
    with open(uploaded_file.name, 'wb') as f:
    # with open(uploaded_filename_new, 'wb') as f:
        f.write(uploaded_file.getvalue())

    # Get the filename of the uploaded file
    uploaded_filename = uploaded_file.name

    # tell whether the file is a `.mat` file
    if not uploaded_filename.endswith('.mat'):
        st.write(f"Please upload a `.mat` file")        
    else:
        # Show analysis options
        # Target size: could be integer like 128, float like 128.0, or tuple like (128, 128) or (128.0, 128.0); default is '(128,128)'
        # Get raw input from user
        # target_size_str = st.text_input('Target size', value='(128,128)', help='The target size of the images')
        # # Parse the input to tuple
        # try:
        #     target_size = eval(target_size_str)
        #     if type(target_size) not in [int, float, tuple, list]:
        #         st.write(f"Please input a valid target size")
        #         target_size = None
        #     elif type(target_size) in [int, float]:
        #         target_size = (target_size, target_size)
        #     else:
        #         pass
        # except:
        #     st.write(f"Please input a valid target size")
        #     target_size = None
        
        




        # enable the button
        # analysis_button = st.button('Analyze', key='analyze_button', help='Press the button to start analyzing the file')
        if st.session_state.get("analyze_button", False):
            st.session_state.disabled = False
        st.write(f"File uploaded: {uploaded_filename}")
    
    # st.write(type(uploaded_filename))
    # st.write(type(uploaded_file.getvalue()))
    if analysis_button:
        # Analyze the file
        valid_cine_slices = prepare_StrainNet_data_from_SuiteHeart(
            uploaded_filename,
            target_pixel_resolution=target_pixel_resolution,
            target_image_shape=target_size)
        st.write(len(valid_cine_slices))

        # Show Results
        n_valid_cine_slices = len(valid_cine_slices)
        check_frame_idx = 10
        for valid_cine_slice_idx, valid_cine_slice in enumerate(valid_cine_slices):
            # Visualize the current slice
            if dynamic_option == 'Static':                
                # if visualize static frame, use matplotlib to show the static frame
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                vis_lv_epi_contour = valid_cine_slice['cine_lv_epi_contour'][check_frame_idx]# - np.array([cine_slices[check_slice_idx]['cine_cropped_myocardium_masks_bbox'][2], cine_slices[check_slice_idx]['cine_cropped_myocardium_masks_bbox'][0]])
                vis_lv_endo_contour = valid_cine_slice['cine_lv_endo_contour'][check_frame_idx]# - np.array([cine_slices[check_slice_idx]['cine_cropped_myocardium_masks_bbox'][2], cine_slices[check_slice_idx]['cine_cropped_myocardium_masks_bbox'][0]])
                vis_rv_insertion_point = valid_cine_slice['cine_rv_insertion_points']#[check_frame_idx]

                axs[0].imshow(valid_cine_slice['cine_cropped_images'][..., check_frame_idx], cmap='gray')
                axs[0].set_title('cine_cropped_images')
                axs[0].plot(vis_lv_epi_contour[:,0], vis_lv_epi_contour[:,1], 'b')
                axs[0].plot(vis_lv_endo_contour[:,0], vis_lv_endo_contour[:,1], 'r')
                axs[0].plot(vis_rv_insertion_point[0], vis_rv_insertion_point[1], 'rx')
                axs[1].imshow(valid_cine_slice['cine_cropped_myocardium_masks'][..., check_frame_idx], cmap='gray')
                axs[1].set_title('cine_cropped_myocardium_masks')
                axs[2].imshow(valid_cine_slice['cine_cropped_images'][..., check_frame_idx], cmap='gray')
                axs[2].imshow(valid_cine_slice['cine_cropped_myocardium_masks'][..., check_frame_idx], cmap='gray', alpha=0.5)
                axs[2].set_title('cine_cropped_images')

                st.pyplot(fig)
            elif dynamic_option == 'Dynamic':
                # if visualize dynamic gif, first save the frames to a gif file, then use st.image to show the dynamic gif
                # save the frames to a gif file
                vis_rootdir.mkdir(parents=True, exist_ok=True)
                frames = []
                for frame_idx in range(valid_cine_slice['cine_cropped_images'].shape[-1]):
                    frames.append(valid_cine_slice['cine_cropped_images'][..., frame_idx])

                # Save the frames as a GIF
                gif_filename = f'valid-slice-{valid_cine_slice_idx}.gif'
                skio.imsave(str(vis_rootdir / Path(gif_filename)), np.array(frames), duration=0.1)  # Adjust duration as needed

                # Show the dynamic gif
                st.image(str(vis_rootdir / Path(gif_filename)))
            else:
                raise ValueError(f"Invalid dynamic_option: {dynamic_option}")

        


        
        # Enable the download button
        # st.session_state.finished_analysis = True
        # finished_analysis = True
        # if st.session_state.get("download_button", False):
        #     st.session_state.disabled = False
        # download_button = st.button('Download analysis result', key='download_button', help='Press the button to download the analysis result')
        # if download_button:
        # st.markdown(f'<a href="{zip_base_filename}.zip" download="{zip_base_filename}.zip">Click here to download the analysis result</a>', unsafe_allow_html=True)

        # Save each of the analysis results to a .mat file
        saved_filenames = []            
        for valid_cine_slice_idx, valid_cine_slice in enumerate(valid_cine_slices):
            # make a subdir for each slice based on upload filename + slice index
            slice_subdir = slice_rootdir / Path(f"{uploaded_filename}_{valid_cine_slice_idx}")
            slice_subdir.mkdir(parents=True, exist_ok=True)
            
            # save the slice to "input.mat" file
            slice_input_filename = slice_subdir / "input.mat"
            slice_input_data = copy.deepcopy(valid_cine_slice)
            slice_input_data['input'] = slice_input_data.pop('cine_cropped_myocardium_masks')
            sio.savemat(slice_input_filename, slice_input_data)
            saved_filenames.append(slice_input_filename)
        
        # zip all the saved files
        zip_base_filename = Path(f"./analyzed_files/zip/{uploaded_filename}")        
        shutil.make_archive(
            base_name = zip_base_filename, 
            format = 'zip', 
            root_dir = slice_rootdir)#,
            # base_dir = slice_rootdir)
        
        with open(f"{zip_base_filename}.zip", "rb") as file:
            download_button = st.download_button(
                label="Download data",
                data=file,
                file_name=f"{zip_base_filename.stem}.zip",
                mime="application/zip",
            )

