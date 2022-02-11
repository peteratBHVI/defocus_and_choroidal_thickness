"""
Author: p.wagner@bhvi.org / p.wagner@unsw.edu.au

Purpose: ease OCT image access and analyses

"""
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objects as go

class TopconSegmentationData:
    # extract thickness data from OCT segmentation file by Topcon Triton data collector

    def __init__(self, oct_data, scan_number):
        self.df = oct_data
        self.scan_number = scan_number
        (self.rnfl,
         self.gclp,
         self.gclpp,
         self.retina,
         self.choroid) = self.find_scan_data(self.df, self.scan_number)
        self.scan_quality = self.get_quality(self.df, self.scan_number)

    @staticmethod
    def get_quality(df, scan_number):
        data_no_col_idx = df.loc[0, :].index[df.loc[0, :].values == 'Data No.'][0]
        idx = df.iloc[:, data_no_col_idx][df.iloc[:, data_no_col_idx].values == str(scan_number)].index[0]
        quality_col_idx = df.loc[0, :].index[df.loc[0, :].values == 'TopQ Image Quality'][0]

        return df.iloc[idx, quality_col_idx]

    @staticmethod
    def find_scan_data(df, scan_number):

        data_no_col_idx = df.loc[0, :].index[df.loc[0, :].values == 'Data No.'][0]
        idx = df.iloc[:, data_no_col_idx][df.iloc[:, data_no_col_idx].values == str(scan_number)].index[0]

        # find rnfl and extract data
        idx_rnfl = df.iloc[idx:, 0][df.iloc[idx:, 0].values == 'RNFL'].index[0]
        rnfl = df.iloc[idx_rnfl + 1: idx_rnfl + 257, 1:].reset_index(drop=True)
        rnfl.columns = [np.arange(0, 512)]
        # find GCL+ aka gclp
        idx_gclp = df.iloc[idx:, 0][df.iloc[idx:, 0].values == 'GCL+'].index[0]
        gclp = df.iloc[idx_gclp + 1: idx_gclp + 257, 1:].reset_index(drop=True)
        gclp.columns = [np.arange(0, 512)]
        # find GCL+ aka gclpp
        idx_gclpp = df.iloc[idx:, 0][df.iloc[idx:, 0].values == 'GCL++'].index[0]
        gclpp = df.iloc[idx_gclpp + 1: idx_gclpp + 257, 1:].reset_index(drop=True)
        gclpp.columns = [np.arange(0, 512)]
        # find Retina
        idx_retina = df.iloc[idx:, 0][df.iloc[idx:, 0].values == 'Retina'].index[0]
        retina = df.iloc[idx_retina + 1: idx_retina + 257, 1:].reset_index(drop=True)
        retina.columns = [np.arange(0, 512)]
        # find Choroid
        idx_choroid = df.iloc[idx:, 0][df.iloc[idx:, 0].values == 'Choroid'].index[0]
        choroid = df.iloc[idx_choroid + 1: idx_choroid + 257, 1:].reset_index(drop=True)
        choroid.columns = [np.arange(0, 512)]

        return rnfl, gclp, gclpp, retina, choroid

    @staticmethod
    def create_averaging_figures(mean, std, max_data, min_data, px_id, quality, scan_type, scan_ids, fp):
        # creating figure choroid thickness figures
        # direct view
        im_output_fn_d = (scan_type.split('_')[-1] + '_' + scan_type.split('_')[-2] +
                          '_mean_direct_view_choroid_thickness.png')
        # # display choroid values
        fig1 = go.Figure(data=[go.Surface(z=mean)])
        fig1_title_name = ('PX_id: ' + str(px_id) + ', Scan Nr: ' + str(scan_ids) +
                           ', Quality: ' + str(quality))
        fig1.update_layout(title=fig1_title_name, autosize=False,
                           scene_camera_eye=dict(x=0.5, y=-10, z=-20),
                           width=900, height=900,
                           margin=dict(l=50, r=50, b=50, t=50))
        fig1.update_layout(scene=dict(xaxis=dict(nticks=10, range=[0, 512], ),
                                      yaxis=dict(nticks=10, range=[0, 256], ),
                                      zaxis=dict(nticks=10, range=[0, 500], ), ),
                           scene_aspectmode='manual',
                           scene_aspectratio=dict(x=12, y=9, z=10)
                           )

        fig1.write_image(os.path.join(fp, 'images', im_output_fn_d))

        im_output_fn_i = (scan_type.split('_')[-1] + '_' + scan_type.split('_')[-2] +
                          '_mean_indirect_view_hist_minmax_choroid_thickness.png')

        # indirect view
        fig2 = go.Figure(data=[go.Surface(z=mean)])
        fig2_title_name = ('PX_id: ' + str(px_id) + ', Scan Nr: ' + str(scan_ids) +
                           ', Quality: ' + str(quality))
        fig2.update_layout(title=fig2_title_name, autosize=False,
                           scene_camera_eye=dict(x=0.5, y=-10, z=20),
                           width=900, height=900,
                           margin=dict(l=50, r=50, b=50, t=50))
        fig2.update_layout(scene=dict(xaxis=dict(nticks=10, range=[0, 512], ),
                                      yaxis=dict(nticks=10, range=[0, 256], ),
                                      zaxis=dict(nticks=10, range=[0, 500], ), ),
                           scene_aspectmode='manual',
                           scene_aspectratio=dict(x=12, y=9, z=10)
                           )

        fig2.write_image(os.path.join(fp, 'images', im_output_fn_i))

        # std of delta
        fig3_output_fn = scan_type.split('_')[-1] + '_' + scan_type.split('_')[-2] + '_std_choroid_thickness.png'
        # # display choroid values
        fig3 = go.Figure(data=[go.Surface(z=std)])
        fig3_title_name = ('PX_id: ' + str(px_id) + ', Scan Nr: ' + str(scan_ids) +
                      ', Quality: ' + str(quality))
        fig3.update_layout(title=fig3_title_name, autosize=False,
                          scene_camera_eye=dict(x=0, y=-10, z=20),
                          width=900, height=900,
                          margin=dict(l=50, r=50, b=50, t=50))
        fig3.update_layout(scene=dict(xaxis=dict(nticks=10, range=[0, 512], ),
                                     yaxis=dict(nticks=10, range=[0, 256], ),
                                     zaxis=dict(nticks=10, range=[0, 100], ), ),
                           scene_aspectmode='manual',
                           scene_aspectratio=dict(x=12, y=9, z=10)
                           )
        fig3.write_image(os.path.join(fp, 'images', fig3_output_fn))
        # hist of fig 3
        fig4_output_fn = scan_type.split('_')[-1] + '_' + scan_type.split('_')[
            -2] + '_hist_std_choroid_thickness.png'
        disp_data_all_choroids_std = std.flatten()
        disp_data_all_choroids_std[disp_data_all_choroids_std > 40] = 40
        fig4 = plt.figure(figsize=(14, 5), dpi=80)
        fig4 = plt.hist(disp_data_all_choroids_std, 40)
        fig4 = plt.title('Std; mean: ' + str(np.round(np.nanmean(std), 4)) + ', std: ' +
                          str(np.round(np.nanstd(std), 4)), fontsize=30)

        plt.savefig(os.path.join(fp, 'images', fig4_output_fn))

        fig5_output_fn = scan_type.split('_')[-1] + '_' + scan_type.split('_')[
            -2] + '_minmax_choroid_thickness.png'
        # # display choroid values
        fig5 = go.Figure(data=[go.Surface(z=(max_data - min_data))])
        fig5_title_name = ('PX_id: ' + ', max-min values ' + str(scan_ids) +
                           ', Quality: ' + str(quality))
        fig5.update_layout(title=fig5_title_name, autosize=False,
                           scene_camera_eye=dict(x=0, y=-10, z=20),
                           width=900, height=900,
                           margin=dict(l=50, r=50, b=50, t=50))

        fig5.update_layout(scene=dict(xaxis=dict(nticks=10, range=[0, 512], ),
                                      yaxis=dict(nticks=10, range=[0, 256], ),
                                      zaxis=dict(nticks=10, range=[0, 200], ), ),
                           scene_aspectmode='manual',
                           scene_aspectratio=dict(x=12, y=9, z=10)
                           )

        fig5.write_image(os.path.join(fp, 'images', fig5_output_fn))

        fig6_output_fn = scan_type.split('_')[-1] + '_' + scan_type.split('_')[-2] + \
                         '_hist_minmax_choroid_thickness.png'

        min_max = max_data - min_data
        min_max = min_max.flatten()
        min_max[min_max > 40] = 40

        fig6 = plt.figure(figsize=(14, 5), dpi=80)
        fig6 = plt.hist(min_max, 40)
        fig6 = plt.title('Max - min values mean: ' + str(np.round(np.nanmean(min_max), 4)) + ', std: ' +
                         str(np.round(np.nanstd(min_max), 4)), fontsize=30)

        plt.savefig(os.path.join(fp, 'images', fig6_output_fn))

        # creat one figure
        fig_sum = plt.figure(figsize=(16, 20), dpi=300, facecolor='blue', edgecolor='k')

        grid = plt.GridSpec(20, 16, wspace=0, hspace=0)
        # From this we can specify subplot locations and extents using the familiary Python slicing syntax:

        fig1_sum = plt.subplot(grid[0:8, 0:8])

        fig1_sum = plt.axis('off')
        fig1_sum = plt.imshow(mpimg.imread(os.path.join(fp, 'images', im_output_fn_d)))


        fig2_sum = plt.subplot(grid[0:8, 8:16])

        fig2_sum = plt.axis('off')
        fig2_sum = plt.imshow(mpimg.imread(os.path.join(fp, 'images', im_output_fn_i)))


        fig3_sum = plt.subplot(grid[8:16, 0:8])

        fig3_sum = plt.axis('off')
        fig3_sum= plt.imshow(mpimg.imread(os.path.join(fp, 'images', scan_type.split('_')[-1] + '_' +
                                                       scan_type.split('_')[-2] +'_std_choroid_thickness.png')))



        fig4_sum = plt.subplot(grid[8:16, 8:16])

        fig4_sum = plt.axis('off')
        fig4_sum = plt.imshow(mpimg.imread(os.path.join(fp, 'images', scan_type.split('_')[-1] + '_' +
                                                        scan_type.split('_')[-2] + '_minmax_choroid_thickness.png')))


        fig5_sum = plt.subplot(grid[16:19, 0:8])

        fig5_sum = plt.axis('off')
        fig5_sum = plt.imshow(mpimg.imread(os.path.join(fp, 'images', scan_type.split('_')[-1] + '_' +
                                                        scan_type.split('_')[-2] + '_hist_std_choroid_thickness.png')))



        fig6_sum = plt.subplot(grid[16:19, 8:16])

        fig6_sum = plt.axis('off')
        fig6_sum = plt.imshow(mpimg.imread(os.path.join(fp, 'images', scan_type.split('_')[-1] + '_' +
                                                        scan_type.split('_')[-2] + '_hist_minmax_choroid_thickness.png')))

        sum_output_fn = scan_type.split('_')[-1] + '_' + scan_type.split('_')[-2] + '_summery_reg.png'

        fig_sum.savefig(os.path.join(fp, sum_output_fn))

    @staticmethod
    def create_rnfl_avg_figures(mean_thickness, px_id, quality, scan_type, scan_ids, fp, fp_oct):
        # creating figure choroid thickness figures
        # direct view
        im_output_fn_d = (str(px_id) + scan_type.split('_')[-1] + '_' + scan_type.split('_')[-2] +
                          '_rnfl_thickness_all.png')
        # # display choroid values
        fig1 = go.Figure(data=[go.Surface(z=abs(-mean_thickness))])
        title_name = ('PX_id: ' + str(px_id) + ', Scan Nr: ' + str(scan_ids) +
                      ', Quality: ' + str(quality))
        fig1.update_layout(title=title_name, autosize=False,
                           scene_camera_eye=dict(x=0.1, y=-10, z=20),
                           width=900, height=900,
                           margin=dict(l=50, r=50, b=50, t=50))
        fig1.update_layout(scene=dict(xaxis=dict(nticks=10, range=[0, 512], ),
                                      yaxis=dict(nticks=10, range=[0, 256], ),
                                      zaxis=dict(nticks=10, range=[0, 300], ), ),
                           scene_aspectmode='manual',
                           scene_aspectratio=dict(x=12, y=9, z=10)
                           )

        fig1.write_image(os.path.join(fp_oct, 'images\\rnfl', im_output_fn_d))
        fig1.write_image(os.path.join(fp, 'images', im_output_fn_d))

        im_output_fn_i = (str(px_id) + scan_type.split('_')[-1] + '_' + scan_type.split('_')[-2] +
                          '_rnfl_thickness_macula.png')

        # indirect view
        fig2 = go.Figure(data=[go.Surface(z=abs(-mean_thickness[50:200, 175:325]))])
        title_name = ('PX_id: ' + str(px_id) + ', Scan Nr: ' + str(scan_ids) +
                      ', Quality: ' + str(quality))
        fig2.update_layout(title=title_name, autosize=False,
                           scene_camera_eye=dict(x=0.1, y=-3, z=10),
                           width=900, height=900,
                           margin=dict(l=50, r=50, b=50, t=50))
        fig2.update_layout(scene=dict(xaxis=dict(nticks=10, range=[0, 150], ),
                                      yaxis=dict(nticks=10, range=[0, 150], ),
                                      zaxis=dict(nticks=10, range=[-1, 150], ), ),
                           scene_aspectmode='manual',
                           scene_aspectratio=dict(x=9, y=9, z=10)
                           )

        fig2.write_image(os.path.join(fp_oct, 'images\\rnfl', im_output_fn_i))
        fig2.write_image(os.path.join(fp, 'images', im_output_fn_i))

    @staticmethod
    def macula_pos_from_rnfl(fp_fn_logbook, px_ids, scan_types, path_oct):
        px_meta = OctDataAccess(fp_fn_logbook, px_ids, scan_types, path_oct)

        columns_names = ['px_id', 'scan_type', 'macula_row_pos', 'macula_col_pos']
        order_test_meta = pd.DataFrame(columns=columns_names)

        for idx, px_id, in enumerate(px_ids):
            for scan_type in scan_types:
                for fn in glob.glob(os.path.join(px_meta.subject_rec_fp[idx],
                                                 scan_type.split('_')[-1] + '_' +
                                                 scan_type.split('_')[1] +
                                                 '*rnfl_mean_reg.csv')):
                    # print(fn)
                    df_new = pd.DataFrame(columns=columns_names)
                    rnfl_data = pd.read_csv(fn, index_col=0)
                    # look in interval [50:200, 175:325] for 0 pos or smaler xx for macula position
                    rnfl_macula = rnfl_data.iloc[50:200, 175:325].values
                    # calculate centre position
                    # horizontal positioning == column space counted from the top left
                    row_pos = np.mean(np.where(rnfl_macula < 1)[0] + 50)
                    # vertical positionng == row space
                    col_pos = np.mean(np.where(rnfl_macula < 1)[1] + 175)

                    rnfl_macula_idxs = (np.where(rnfl_macula < 1)[1] + 175), (256 + -(np.where(rnfl_macula < 1)[0] + 50))
                    df_new['px_id'] = [px_id]
                    df_new['scan_type'] = scan_type
                    df_new['macula_row_pos'] = row_pos
                    df_new['macula_col_pos'] = col_pos
                    order_test_meta = order_test_meta.append(df_new)

        # calculating residual of center positions for each eye across pre and post intervention
        order_test_meta.loc[:, 'col_dev'] = np.round(order_test_meta.loc[:, 'macula_col_pos'] -
                                                     np.mean(order_test_meta.loc[:, 'macula_col_pos']), 0)

        order_test_meta.loc[:, 'row_dev'] = np.round(order_test_meta.loc[:, 'macula_row_pos'] -
                                                     np.mean(order_test_meta.loc[:, 'macula_row_pos']), 0)

        return order_test_meta.reset_index(drop=True)

    @staticmethod
    def average_with_referece_to_macula(macula_pos_all, scan_type, layer_id):
        # calculate OD initial scan average with adjustment for macula positioning
        eye_id = scan_type.split('_')[2]
        scan_set = scan_type.split('_')[1]
        macula_pos_set = macula_pos_all.loc[(macula_pos_all.eye_id == eye_id) &
                                         (macula_pos_all.scan_type == scan_set), :].reset_index(drop=True)
        layer_fn = '_'.join([eye_id, scan_set, layer_id, 'mean_reg.csv'])
        layer_all = []
        for idx, layer_fp in enumerate(macula_pos_set.fp):
            pos_adjust_layer = np.empty((256, 512), dtype=float)
            pos_adjust_layer[:] = np.nan

            layer = pd.read_csv(os.path.join(layer_fp, layer_fn), index_col=0).values.astype(float)
            # layer[layer < 100] = np.nan
            row_pos = int(macula_pos_set.loc[idx, 'row_dev'])
            col_pos = int(macula_pos_set.loc[idx, 'col_dev'])

            print(idx, ' ', macula_pos_set.loc[idx, 'row_dev'], ' ', macula_pos_set.loc[idx, 'col_dev'])
            if (row_pos <= 0) & (col_pos <= 0):
                pos_adjust_layer[- row_pos: 256, - col_pos:512] = layer[0:256 + row_pos, 0:512 + col_pos]
            #         print ('< <')
            elif (row_pos <= 0) & (col_pos >= 0):
                pos_adjust_layer[- row_pos: 256, 0:512 - col_pos] = layer[0:256 + row_pos, col_pos:512]
            #         print( '< > ')
            elif (row_pos >= 0) & (col_pos <= 0):
                pos_adjust_layer[0: 256 - row_pos, - col_pos:512] = layer[row_pos:256, 0: 512 + col_pos]
            #         print( '> < ')
            elif (row_pos >= 0) & (col_pos >= 0):
                pos_adjust_layer[0:256 - row_pos, 0:512 - col_pos] = layer[row_pos: 256, col_pos: 512]
            #         print( '> > ')
            #     else:
            #         pos_adjust_rnfl = rnfl
            #         print( '= =')

            layer_all.append(pos_adjust_layer)

        layer_all = np.array(layer_all)
        mean_all = np.nanmean(layer_all, axis=0)
        std_all = np.nanstd(layer_all, axis=0)
        max_all = np.nanmax(layer_all, axis=0)
        min_all = np.nanmin(layer_all, axis=0)

        return mean_all, std_all, max_all, min_all

class OctDataAccess:

    def __init__(self, fp_fn_logbook, px_ids, scan_types, path_oct):
        self.fp_fn_logbook = fp_fn_logbook
        self.px_ids = px_ids
        self.scan_types = scan_types
        self.path_oct = path_oct
        self.log_master = pd.read_excel(self.fp_fn_logbook, sheet_name='master', index_col=1, engine='xlrd')
        self.subject_ids = self.get_subject_ids(self.log_master, self.px_ids)
        self.subject_rec_fp = self.get_subjects_rec_fp(self.subject_ids)
        [self.oct_scans_fp, self.oct_scans_ids] = self.get_oct_scans_fp_ids()

    @staticmethod
    def get_quality(df, scan_number):
        data_no_col_idx = df.loc[0, :].index[df.loc[0, :].values == 'Data No.'][0]
        idx = df.iloc[:, data_no_col_idx][df.iloc[:, data_no_col_idx].values == str(scan_number)].index[0]
        quality_col_idx = df.loc[0, :].index[df.loc[0, :].values == 'TopQ Image Quality'][0]
        return df.iloc[idx, quality_col_idx]

    @staticmethod
    def get_scan_time(df, scan_number):
        data_no_col_idx = df.loc[0, :].index[df.loc[0, :].values == 'Data No.'][0]
        idx = df.iloc[:, data_no_col_idx][df.iloc[:, data_no_col_idx].values == str(scan_number)].index[0]
        capture_date_col_idx = df.loc[0, :].index[df.loc[0, :].values == 'Capture Date'][0]
        capture_time_col_idx = df.loc[0, :].index[df.loc[0, :].values == 'Capture Time'][0]

        date_time = df.iloc[idx, capture_date_col_idx].replace(" ", "") + '/' + df.iloc[idx, capture_time_col_idx]
        return date_time

    def get_subject_ids(self, log_master, px_ids):
        subject_ids_col_names = []
        for px_id in px_ids:
            for col_name in self.log_master.columns:
                if self.log_master.loc['px_id', col_name] == px_id:
                    subject_ids_col_names.append(col_name)

        subject_ids = log_master.loc['subject', subject_ids_col_names].reset_index(drop=True)
        return subject_ids

    def get_subjects_rec_fp(self, subject_ids):
        rec_fp = list()
        # find all in folders for participants
        for subject_id in subject_ids:
            fp = os.path.join(self.path_oct, subject_id)

            if os.path.isdir(fp):
                rec_fp.append(fp)
            else:
                print(fp, ' not recoreded')
        return rec_fp

    def get_oct_scans_fp_ids(self):
        octs_fp = []
        oct_scans_ids = []
        for px_id in self.px_ids:
            for col_name in self.log_master.columns:
                if self.log_master.loc['px_id', col_name] == px_id:

                    subject_id = self.log_master.loc['subject', col_name]
                    # print(col_name, subject_id, 'test')
                    for scan_type in self.scan_types:
                        # find scan numbers
                        scan_nrs = self.log_master.loc[scan_type, col_name].split(',')
                        # get rid of spaces
                        scan_nrs = [x.strip() for x in scan_nrs]
                        # combine subject file path with scan number
                        for scan_nr in scan_nrs:
                            fp = os.path.join(self.path_oct, subject_id, scan_nr)
                            if os.path.isdir(fp):
                                octs_fp.append(fp)
                                oct_scans_ids.append(scan_nr)
                            else:
                                print(fp, ' no OCT filepath found')
        return octs_fp, oct_scans_ids

class FFT_filtering:
    @staticmethod
    def apply_radial_mask(im_fft, radius=0, mask_value=0):

        # pass filter high low whatever
        rows, cols = im_fft.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        if mask_value == 1:
            mask = np.zeros((rows, cols), np.uint8)
        if mask_value == 0:
            mask = np.ones((rows, cols), np.uint8)

        centre = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - centre[0]) ** 2 + (y - centre[1]) ** 2 <= radius ** 2
        mask[mask_area] = mask_value

        im_fft_masked = im_fft * mask

        return im_fft_masked

    @staticmethod
    def apply_fft_with_radial_mask(c_map, radius, mask_value, cut_off_threshold):
        from scipy import fftpack
        c_map_fft = fftpack.fft2(c_map)

        # maks - within radius, 0 disregard, 1 keep
        im_fft_masked = FFT_filtering.apply_radial_mask(c_map_fft, radius=radius, mask_value=mask_value)

        # Reconstruct the denoised image from the filtered spectrum, keep only the
        # real part for display.
        c_map_fft_filtered = fftpack.ifft2(im_fft_masked).real
        # set limits for plotting image
        # c_map_fft_filtered[c_map_fft_filtered > cut_off_threshold] = cut_off_threshold
        # c_map_fft_filtered[c_map_fft_filtered < -cut_off_threshold] = -cut_off_threshold
        return c_map_fft_filtered