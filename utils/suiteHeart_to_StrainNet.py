from utils.suiteHeart_utils import get_SuiteHeart_dict_myocardium_rescaled_contours_and_masks, get_SuiteHeart_dict_resized_and_cropped_images
from utils.suiteHeart_utils import get_SuiteHeart_dict_myocardium_masks, get_SuiteHeart_dict_cropped_images
from utils.loadmat import loadmat
import numpy as np

def prepare_StrainNet_data_from_SuiteHeart(mat_filename, target_pixel_resolution = 'ori', target_image_shape = (128,128)):
    # target_pixel_resolution could be 'ori' or '1x1' or arbitary float number
    cine_mat = loadmat(mat_filename)
    if isinstance(target_pixel_resolution, str) and target_pixel_resolution == 'ori':
        cine_lv_endo_contours_all, cine_lv_epi_contours_all, cine_myo_masks_all, cine_myocardium_masks_bbox_all, cine_rv_insertion_points_all  = \
            get_SuiteHeart_dict_myocardium_masks(
                cine_mat, 
                ori_image_shape=cine_mat['raw_image'][0,0].shape if isinstance(cine_mat['raw_image'], np.ndarray) else cine_mat['raw_image'][0].shape,
                target_image_shape=target_image_shape, 
                centering=False, 
                check_slice_indices=None,
                force_keep_slice_dim=True)
        cine_cropped_images, cine_cropped_images_bbox = \
            get_SuiteHeart_dict_cropped_images(
                cine_mat, 
                target_image_shape=target_image_shape,
                force_keep_slice_dim=True)
    elif isinstance(target_pixel_resolution, str) and target_pixel_resolution == '1x1':
        cine_lv_endo_contours_all, cine_lv_epi_contours_all, cine_myo_masks_all, cine_rv_insertion_points_all  = \
            get_SuiteHeart_dict_myocardium_rescaled_contours_and_masks(
                cine_mat, 
                target_image_shape=target_image_shape, 
                centering=False, 
                check_slice_indices=None,
                force_keep_slice_dim=True)
        # cine_cropped_images, cine_cropped_images_bbox = \
        cine_cropped_images = \
            get_SuiteHeart_dict_resized_and_cropped_images(
                cine_mat, 
                target_image_shape=target_image_shape,
                force_keep_slice_dim=True)
    elif isinstance(target_pixel_resolution, float) or isinstance(target_pixel_resolution, int):
        if isinstance(cine_mat['pixel_size'], list):
            cine_mat_pixel_spacing = cine_mat['pixel_size'][0][0]
        else:
            cine_mat_pixel_spacing = cine_mat['pixel_size'][0,0][0]
        cine_lv_endo_contours_all, cine_lv_epi_contours_all, cine_myo_masks_all, cine_rv_insertion_points_all  = \
            get_SuiteHeart_dict_myocardium_rescaled_contours_and_masks(
                cine_mat, 
                target_image_shape=target_image_shape, 
                centering=False, 
                check_slice_indices=None, 
                rescale_method='rescale', 
                rescale_ratio=cine_mat_pixel_spacing/target_pixel_resolution,
                force_keep_slice_dim=True)    
        # cine_cropped_images, cine_cropped_images_bbox = \
        cine_cropped_images = \
            get_SuiteHeart_dict_resized_and_cropped_images(
                cine_mat, 
                target_image_shape=target_image_shape, 
                rescale_method='rescale', 
                rescale_ratio=cine_mat_pixel_spacing/target_pixel_resolution,
                force_keep_slice_dim=True)
    else:
        raise ValueError('Invalid target_pixel_resolution')
    
    
    cine_valid_slice_indices = np.where(np.array([np.sum(cine_myo_masks_all[slice_idx]) for slice_idx in range(len(cine_myo_masks_all))]) > 0)[0]

    cine_valid_slices = []
    for cine_slice_idx in cine_valid_slice_indices:
        cine_slice = {
            'patient_name': cine_mat['patient_name'],
            'cine_slice_idx': cine_slice_idx,
            'cine_slice_name': f'{cine_mat["patient_name"]}-{cine_slice_idx}',
            'cine_slice_mat_filename': str(mat_filename),
            'cine_slice_has_contour': True,
        }
        cine_slice['cine_lv_epi_contour'] = cine_lv_epi_contours_all[:, cine_slice_idx]
        cine_slice['cine_lv_endo_contour'] = cine_lv_endo_contours_all[:, cine_slice_idx]

        cine_slice['cine_cropped_images'] = cine_cropped_images[cine_slice_idx]
        # cine_slice['cine_cropped_images_bbox'] = cine_cropped_images_bbox[cine_slice_idx]

        cine_slice['cine_cropped_myocardium_masks'] = cine_myo_masks_all[cine_slice_idx]            
        # cine_slice['cine_cropped_myocardium_masks_bbox'] = cine_myocardium_masks_bbox_all[cine_slice_idx]

        cine_slice['cine_rv_insertion_points'] = cine_rv_insertion_points_all[cine_slice_idx]

        if 'trigger_time' in cine_mat.keys():
            if isinstance(cine_mat['trigger_time'], list):
                cine_slice['cine_frame_time_stamps'] = np.array(cine_mat['trigger_time'])
            else:
                cine_slice['cine_frame_time_stamps'] = cine_mat['trigger_time'][:,cine_slice_idx]
        #     cine_slice['cine_frame_time_stamps'] = cine_mat['trigger_time'][:,cine_slice_idx]

        cine_valid_slices.append(cine_slice)
    
    return cine_valid_slices