'''
Created on Aug 9, 2016

@author: Zongyang Li
'''
import cv2
import os,sys,argparse, shutil, terra_common, requests
import numpy as np
from glob import glob
from plyfile import PlyData, PlyElement
from datetime import date, timedelta
from terrautils import betydb
import multiprocessing
import analysis_utils

PLOT_RANGE_NUM = 54
PLOT_COL_NUM = 32
HIST_BIN_NUM = 500

# [(35th percentile from E + 35th percentile from W) / 2] cm * 0.97841 + 25.678cm
B_TH_PERC = 0.35
B_F_SLOPE = 0.97841
B_F_OFFSET = 25.678

# [37th percentile from E sensor] cm * 0.94323 + 26.41cm
E_TH_PERC = 0.37
E_F_SLOPE = 0.94323
E_F_OFFSET = 26.41

# [41st percentile from W sensor] cm * 0.98132 + 24.852cm
W_TH_PERC = 0.41
W_F_SLOPE = 0.98132
W_F_OFFSET = 24.852

def options():
    
    parser = argparse.ArgumentParser(description='Height Distribution Extractor in Roger',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-m", "--mode", help="all day flag, all for all day process, given parent directory as input, one for one day process")
    parser.add_argument("-p", "--ply_dir", help="ply directory")
    parser.add_argument("-j", "--json_dir", help="json directory")
    parser.add_argument("-o", "--out_dir", help="output directory")
    parser.add_argument("-y", "--year", help="specify which year to process")
    parser.add_argument("-d", "--start_month", help="specify start month")
    parser.add_argument("-f", "--end_month", help="specify end month")
    parser.add_argument("-s", "--start_date", help="specify start date")
    parser.add_argument("-e", "--end_date", help="specify end date")
    
    args = parser.parse_args()

    return args


def main():
    print("start...")
    
    args = options()
    
    if args.mode == 'all':
        process_all_scanner_data(args.ply_dir, args.json_dir, args.out_dir)

    if args.mode == 'one':
        full_day_gen_hist(args.ply_dir, args.json_dir, args.out_dir)
        
    if args.mode == 'date':
        process_by_dates(args.ply_dir, args.json_dir, args.out_dir, args.year, args.start_month, args.end_month, args.start_date, args.end_date)
        
    
    in_dir = os.path.join(args.out_dir, 'downSamplingMedial')
    out_dir = os.path.join(in_dir, 'outCsv')
    for sensor_d in ('e', 'w'):
        for gridSize in (0,2,4,8,10,15):
            for i in range(4):
                analysis_utils.create_plot_height_histogram(in_dir, out_dir, i, gridSize*2, sensor_d)
                
                analysis_utils.create_sub_plot_height_histogram(in_dir, os.path.join(out_dir, 'subPlot'), gridSize*2, sensor_d)
    
    return

def process_all_scanner_data(ply_parent, json_parent, out_parent):
    
    list_dirs = os.listdir( ply_parent )
    
    for str_date in list_dirs:
        print(str_date)
        ply_path = os.path.join(ply_parent, str_date)
        json_path = os.path.join(json_parent, str_date)
        out_path = os.path.join(out_parent, str_date)
        if not os.path.isdir(ply_path):
            print(ply_path)
            continue
        if not os.path.isdir(json_path):
            print(json_path)
            continue
    
        print('start processing ' + str_date)
        convt = terra_common.CoordinateConverter()
        try:
            q_flag = convt.bety_query(str_date, True)
            if not q_flag:
                print('Bety query failed')
                continue
            
            full_day_gen_hist_multi_process(ply_path, json_path, out_path, convt)
            
            full_day_array_integrate_s6(out_path, convt)
    
        except Exception as ex:
            fail(str(ex))

    
    return

def process_by_dates(ply_parent, json_parent, out_parent, str_year, str_start_month, str_end_month, str_start_date, str_end_date):
    
    d1 = date(int(str_year), int(str_start_month), int(str_start_date))
    d2 = date(int(str_year), int(str_end_month), int(str_end_date))
    
    deltaDay = d2 - d1
    
    for i in range(deltaDay.days+1):
        str_date = str(d1+timedelta(days=i))
        print(str_date)
        ply_path = os.path.join(ply_parent, str_date)
        json_path = os.path.join(json_parent, str_date)
        out_path = os.path.join(out_parent, str_date)
        convt = terra_common.CoordinateConverter()
        if not os.path.isdir(ply_path):
            print('ply miss'+ply_path)
            continue
        if not os.path.isdir(json_path):
            print('json miss'+json_path)
            continue
        try:
            print('bety query')
            q_flag = convt.bety_query(str_date, True)
            if not q_flag:
                print('Bety query failed')
                continue
            
            print('multi process')
            full_day_gen_hist_multi_process(ply_path, json_path, out_path, convt)
    
            full_day_array_integrate_s6(out_path, convt)
            
        except Exception as ex:
            fail(str_date + str(ex))
    
    return

def process_one_month_data(ply_parent, json_parent, out_parent, str_year, str_month, str_start_date, str_end_date):
    
    for day in range(int(str_start_date), int(str_end_date)+1):
        target_date = date(int(str_year), int(str_month), day)
        str_date = target_date.isoformat()
        print(str_date)
        ply_path = os.path.join(ply_parent, str_date)
        json_path = os.path.join(json_parent, str_date)
        out_path = os.path.join(out_parent, str_date)
        convt = terra_common.CoordinateConverter()
        if not os.path.isdir(ply_path):
            print('ply miss'+ply_path)
            continue
        if not os.path.isdir(json_path):
            print('json miss'+json_path)
            continue
        try:
            q_flag = convt.bety_query(str_date, True)
            if not q_flag:
                print('Bety query failed')
                continue
            
            full_day_gen_hist(ply_path, json_path, out_path, convt)
    
            full_day_array_integrate_s6(out_path)
            
            #insert_height_traits_into_betydb(out_path, out_path, str_date, B_TH_PERC, 'e')
        except Exception as ex:
            fail(str_date + str(ex))
    
    return


def full_day_gen_hist(ply_path, json_path, out_path, convt):
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    list_dirs = os.walk(ply_path)
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            print("Start processing "+ d)
            p_path = os.path.join(ply_path, d)
            j_path = os.path.join(json_path, d)
            o_path = os.path.join(out_path, d)
            if not os.path.isdir(p_path):
                continue
            
            if not os.path.isdir(j_path):
                continue
            
            gen_hist_s6_analysis(p_path, j_path, o_path, convt)
    
    return

def full_day_gen_hist_multi_process(ply_path, json_path, out_path, convt):
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    list_dirs = [os.path.join(ply_path,o) for o in os.listdir(ply_path) if os.path.isdir(os.path.join(ply_path,o))]
    json_dirs = [os.path.join(json_path,o) for o in os.listdir(ply_path) if os.path.isdir(os.path.join(ply_path,o))]
    out_dirs = [os.path.join(out_path,o) for o in os.listdir(ply_path) if os.path.isdir(os.path.join(ply_path,o))]
    numDirs = len(list_dirs)
    
    print ("Starting ply to npy conversion...")
    pool = multiprocessing.Pool()
    NUM_THREADS = min(8,numDirs)
    print('numDirs:{}   NUM_THREADS:{}'.format(numDirs, NUM_THREADS))
    for cpu in range(NUM_THREADS):
        pool.apply_async(ply_to_npy, [list_dirs[cpu::NUM_THREADS], json_dirs[cpu::NUM_THREADS], out_dirs[cpu::NUM_THREADS], convt])
    pool.close()
    pool.join()
    print ("Completed ply to npy conversion...")
    
    return

def ply_to_npy(ply_dirs, json_dirs, out_dirs, convt):
    for p, j, o in zip(ply_dirs, json_dirs, out_dirs):
        #Generate jpgs and geoTIFs from .bin
        try:
            gen_hist_s6_analysis(p, j, o, convt)
            #bin_to_geotiff.stereo_test(s, s)
        except Exception as ex:
            fail("\tFailed to process folder %s: %s" % (p, str(ex)))
    

# 25%/50%/75%/100%/subplot divisions plus 4/8/16/20/30 grid size down-sampling
def gen_hist_s6_analysis(ply_path, json_path, out_dir, convt):
    
    baseName = os.path.basename(ply_path)
    print(baseName)
    
    if not os.path.isdir(ply_path):
        fail('Could not find input directory: ' + ply_path)
        
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    # parse json file
    metas, ply_file_wests, ply_file_easts, gImgWests, pImgWests, gImgEasts, pImgEasts = analysis_utils.find_input_files_all(ply_path, json_path)
    
    for meta, ply_file_west, ply_file_east, gImgWest, pImgWest, gImgEast, pImgEast in zip(metas, ply_file_wests, ply_file_easts, gImgWests, pImgWests, gImgEasts, pImgEasts):
        json_dst = os.path.join(out_dir, meta)
        if os.path.exists(json_dst):
            return
        
        metadata = analysis_utils.lower_keys(analysis_utils.load_json(os.path.join(json_path, meta))) # make all our keys lowercase since keys appear to change case (???)
        
        center_position = analysis_utils.get_position(metadata) # (x, y, z) in meters
        scanDirection = analysis_utils.get_direction(metadata) # scan direction
        
        plySrc = PlyData.read(os.path.join(ply_path, ply_file_west))
        pImg = cv2.imread(os.path.join(json_path, pImgWest), -1)
        gImg = cv2.imread(os.path.join(json_path, gImgWest), -1)
        
        print('west')
        for gridSize in (0,2,4,8,10,15):
            plyData = analysis_utils.ply_down_sampling(pImg, gImg, plySrc.elements[0], gridSize, 0)
            hist_w, hist_subplot_w = gen_height_histogram_four_plot_level(plyData, scanDirection, out_dir, 'w', center_position, convt)
        
            histPath = os.path.join(out_dir, 'hist_w_'+str(gridSize*2)+'.npy')
            try:
                np.save(histPath, hist_w)
            except Exception as ex:
                fail(str(ex))
            histPath = os.path.join(out_dir, 'hist_subplot_w_'+str(gridSize*2)+'.npy')
            np.save(histPath, hist_subplot_w)
        
        print("east")
        plySrc = PlyData.read(os.path.join(ply_path, ply_file_east))
        pImg = cv2.imread(os.path.join(json_path, pImgEast), -1)
        gImg = cv2.imread(os.path.join(json_path, gImgEast), -1)
        for gridSize in (0,2,4,8,10,15):
            plyData = analysis_utils.ply_down_sampling(pImg, gImg, plySrc.elements[0], gridSize, 0)
            hist_e, hist_subplot_e = gen_height_histogram_four_plot_level(plyData, scanDirection, out_dir, 'e', center_position, convt)
            
            histPath = os.path.join(out_dir, 'hist_e_'+str(gridSize*2)+'.npy')
            np.save(histPath, hist_e)
            histPath = os.path.join(out_dir, 'hist_subplot_e_'+str(gridSize*2)+'.npy')
            np.save(histPath, hist_subplot_e)
        
        shutil.copyfile(os.path.join(json_path, meta), json_dst)
    
    return

def gen_hist_s4_analysis(ply_path, json_path, out_dir, convt):
    
    print('start processing gen_hist_s4_analysis')
    
    if not os.path.isdir(ply_path):
        fail('Could not find input directory: ' + ply_path)
        
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    # parse json file
    print('parse json file')
    metas, ply_file_wests, ply_file_easts = analysis_utils.find_input_files(ply_path, json_path)
    
    for meta, ply_file_west, ply_file_east in zip(metas, ply_file_wests, ply_file_easts):
        json_dst = os.path.join(out_dir, meta)
        if os.path.exists(json_dst):
            return
        
        metadata = analysis_utils.lower_keys(analysis_utils.load_json(os.path.join(json_path, meta))) # make all our keys lowercase since keys appear to change case (???)
        
        center_position = analysis_utils.get_position(metadata) # (x, y, z) in meters
        scanDirection = analysis_utils.get_direction(metadata) # scan direction
        
        plywest = PlyData.read(os.path.join(ply_path, ply_file_west))
        print('gen_height_histogram_four_plot_level west')
        hist_w, hist_subplot_w = gen_height_histogram_four_plot_level(plywest, scanDirection, out_dir, 'w', center_position, convt)
        
        print('save npy file')
        histPath = os.path.join(out_dir, 'hist_w.npy')
        try:
            np.save(histPath, hist_w)
        except Exception as ex:
            fail(str(ex))
        histPath = os.path.join(out_dir, 'hist_subplot_w.npy')
        np.save(histPath, hist_subplot_w)
        
        plyeast = PlyData.read(os.path.join(ply_path, ply_file_east))
        hist_e, hist_subplot_e = gen_height_histogram_four_plot_level(plyeast, scanDirection, out_dir, 'e', center_position, convt)
        
        histPath = os.path.join(out_dir, 'hist_e.npy')
        np.save(histPath, hist_e)
        histPath = os.path.join(out_dir, 'hist_subplot_e.npy')
        np.save(histPath, hist_subplot_e)
        
        shutil.copyfile(os.path.join(json_path, meta), json_dst)
    
    return

def get_height_result_s6(in_dir, sensor_d, gridSize, convt):
    
    if not os.path.isdir(in_dir):
        fail('Could not find input directory: ' + in_dir)
    
    xRange = -1
    hist_data = np.zeros((4,16,HIST_BIN_NUM))
    hist_subplot_data = np.zeros((32, HIST_BIN_NUM))
    
    # parse json file
    metafile, hist, hist_subplot = analysis_utils.find_result_file_s6(in_dir, sensor_d, gridSize)
    if metafile == [] or hist == [] or hist_subplot == [] :
        return xRange, hist_data, hist_subplot_data
    
    
    metadata = analysis_utils.lower_keys(analysis_utils.load_json(metafile))
    center_position = analysis_utils.get_position(metadata)
    hist_data = np.load(hist, 'r')
    hist_subplot_data = np.load(hist_subplot, 'r')
        
    for i in range(PLOT_RANGE_NUM):
        xmin = convt.np_bounds_subplot[i][0][0]
        xmax = convt.np_bounds_subplot[i][0][1]
        if (center_position[0] > xmin) and (center_position[0] <= xmax):
            xRange = i + 1
            break
    
    return xRange, hist_data, hist_subplot_data


def create_normalization_hist(in_dir, out_dir, convt):
    
    list_dirs = os.walk(in_dir)
    heightHist = np.zeros((PLOT_COL_NUM*PLOT_RANGE_NUM, HIST_BIN_NUM))
    plotScanCount = np.zeros((PLOT_COL_NUM*PLOT_RANGE_NUM))
    
    for root, dirs, files in list_dirs:
        for d in dirs:
            full_path = os.path.join(in_dir, d)
            if not os.path.isdir(full_path):
                continue
            
            plotNum, hist, top = get_height_result_s6(full_path)
            if len(plotNum) < PLOT_COL_NUM:
                continue
            
            for j in range(0,plotNum.size):
                heightHist[plotNum[j]-1] = heightHist[plotNum[j]-1]+hist[j]
                plotScanCount[plotNum[j]-1] = plotScanCount[plotNum[j]-1] + 1
                
    for i in range(0, PLOT_COL_NUM*PLOT_RANGE_NUM):
        if plotScanCount[i] != 0:
            heightHist[i] = heightHist[i]/plotScanCount[i]
    
    histfile = os.path.join(in_dir, 'heightHist.npy')
    np.save(histfile, heightHist)
    
    hist_out_file = os.path.join(in_dir, 'hist.txt')
    np.savetxt(hist_out_file, np.array(heightHist), delimiter="\t")
    
    
    return

def full_day_array_integrate_s6(in_dir, convt):
    
    dir_path = os.path.dirname(in_dir)
    str_date = os.path.basename(in_dir)
    save_dir = os.path.join(dir_path, 'downSamplingMedial')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    for sensor_d in ['e','w']:
        for gridSize in (0,2,4,8,10,15):
            list_dirs = os.walk(in_dir)
            heightHist = np.zeros((4, PLOT_RANGE_NUM, 16, HIST_BIN_NUM))
            heightHist_subplot = np.zeros((PLOT_RANGE_NUM, 32, HIST_BIN_NUM))
            plotScanCount = np.zeros((PLOT_RANGE_NUM,16))
            plotScanCount_subplot = np.zeros((PLOT_RANGE_NUM,32))
            for root, dirs, files in list_dirs:
                for d in dirs:
                    full_path = os.path.join(in_dir, d)
                    if not os.path.isdir(full_path):
                        continue
                    
                    xRange, hist, hist_subplot = get_height_result_s6(full_path, sensor_d, gridSize, convt)
                    if xRange < 0:
                        continue
                    
                    xRange = xRange - 1
                    heightHist[:, xRange, :, :] = heightHist[:, xRange, :, :]+hist
                    heightHist_subplot[xRange, :, :] = heightHist_subplot[xRange, :, :]+hist_subplot
                    
                    plotScanCount[xRange, :] = plotScanCount[xRange, :] + 1
                    plotScanCount_subplot[xRange, :] = plotScanCount_subplot[xRange, :] + 1
                    
            plotShape = plotScanCount.shape
            for x in range(0, plotShape[0]):
                for y in range(0, plotShape[1]):
                    if plotScanCount[x, y] != 0:
                        heightHist[:,x,y] = heightHist[:,x,y]/plotScanCount[x,y]
                        
            plotShape = plotScanCount_subplot.shape
            for x in range(0, plotShape[0]):
                for y in range(0, plotShape[1]):
                    if plotScanCount_subplot[x, y] != 0:
                        heightHist_subplot[x,y] = heightHist_subplot[x,y]/plotScanCount_subplot[x,y]
                        
            
            histfile = os.path.join(in_dir, 'heightHist_'+sensor_d+'_'+str(gridSize*2)+'.npy')
            np.save(histfile, heightHist)
            dst_path = os.path.join(save_dir, str_date + '_heightHist_'+sensor_d+'_'+str(gridSize*2)+'.npy')
            shutil.copyfile(histfile, dst_path)
            
            histfile = os.path.join(in_dir, 'heightHist_subplot_'+sensor_d+'_'+str(gridSize*2)+'.npy')
            np.save(histfile, heightHist_subplot)
            dst_path = os.path.join(save_dir, str_date + '_heightHist_subplot_'+sensor_d+'_'+str(gridSize*2)+'.npy')
            shutil.copyfile(histfile, dst_path)
        
    return


def full_day_array_to_xlsx(in_dir, convt):
    
    for sensor_d in ['e','w']:
        list_dirs = os.walk(in_dir)
        heightHist = np.zeros((PLOT_COL_NUM*PLOT_RANGE_NUM, HIST_BIN_NUM))
        topMat = np.zeros((PLOT_COL_NUM*PLOT_RANGE_NUM))
        for root, dirs, files in list_dirs:
            for d in dirs:
                full_path = os.path.join(in_dir, d)
                if not os.path.isdir(full_path):
                    continue
                
                plotNum, hist, top = get_height_result_s6(in_dir, 'w', 0, convt)
                if len(plotNum) < PLOT_COL_NUM:
                    continue
                
                for j in range(0,plotNum.size):
                    if plotNum[j] == 0:
                        continue
                    
                    heightHist[plotNum[j]-1] = heightHist[plotNum[j]-1]+hist[j]
                    
                    if topMat[plotNum[j]-1] < top[j]:
                        topMat[plotNum[j]-1] = top[j]
        
        histfile = os.path.join(in_dir, 'heightHist_'+sensor_d+'.npy')
        topfile = os.path.join(in_dir, 'topHist_'+sensor_d+'.npy')
        np.save(histfile, heightHist)
        np.save(topfile, topMat)
        '''
        hist_out_file = os.path.join(in_dir, 'hist_'+sensor_d+'.txt')
        np.savetxt(hist_out_file, np.array(heightHist), delimiter="\t")
        top_out_file = os.path.join(in_dir, 'top_'+sensor_d+'.txt')
        np.savetxt(top_out_file, np.array(topMat), delimiter="\t")
        '''
    
    return


def gen_height_histogram_four_plot_level(plydata, scanDirection, out_dir, sensor_d, center_position, convt):
    
    gantry_z_offset = 0.35
    zGround = (3.445 - center_position[2] + gantry_z_offset)*1000
    yRange = 16
    yShift = analysis_utils.offset_choice(scanDirection, sensor_d)
    zOffset = 10
    scaleParam = 1000
    hist = np.zeros((4, yRange, HIST_BIN_NUM))
    hist_subplot = np.zeros((yRange*2, HIST_BIN_NUM))
    data = plydata
    
    xRange = analysis_utils.field_x_2_range(center_position[0],convt)
    
    if data.size == 0:
        print('failed: no available data')
        return hist, hist_subplot
    
    for i in range(2, yRange*2-2):
        ymin = (convt.np_bounds_subplot[xRange][i][2]+yShift) * scaleParam
        ymax = (convt.np_bounds_subplot[xRange][i][3]+yShift) * scaleParam
        specifiedIndex = np.where((data["y"]>ymin) & (data["y"]<ymax))
        target = data[specifiedIndex]
        
        '''
        

        for j in range(0, HIST_BIN_NUM):
            if j == 0:
                zmax = zGround - 50
                zIndex = np.where((target["z"]<zmax))
            else:
                zmin = zGround + (j-6) * zOffset
                zmax = zGround + (j-5) * zOffset
                zIndex = np.where((target["z"]>zmin) & (target["z"]<zmax))
            num = len(zIndex[0])
            hist_subplot[i][j] = num
            '''
            
        
        zVal = (target["z"]-zGround+50)/zOffset
        
        hist_subplot[i] = np.histogram(zVal, bins=range(-1, HIST_BIN_NUM), normed=False)[0]
        
    
    for i in range(1, yRange-1):
        ymin = (convt.np_bounds_subplot[xRange][i*2+1][2]+yShift) * scaleParam
        ymax = (convt.np_bounds_subplot[xRange][i*2][3]+yShift) * scaleParam
        plot_length = ymax - ymin
        
        for p in range(0, 4):
            target_length = plot_length * (float(p+1)/4)
            border_length = (plot_length - target_length)/2
            target_ymin = ymin + border_length
            target_ymax = ymax - border_length
            specifiedIndex = np.where((data["y"]>target_ymin) & (data["y"]<target_ymax))
            target = data[specifiedIndex]
            
            zVal = (target["z"]-zGround+50)/zOffset
        
            hist[p][i] = np.histogram(zVal, bins=range(-1, HIST_BIN_NUM), normed=False)[0]
            '''
            for j in range(0, HIST_BIN_NUM):
                if j == 0:
                    zmax = zGround - 50
                    zIndex = np.where((target["z"]<zmax))
                else:
                    zmin = zGround + (j-6) * zOffset
                    zmax = zGround + (j-5) * zOffset
                    zIndex = np.where((target["z"]>zmin) & (target["z"]<zmax))
                    
                num = len(zIndex[0])
                hist[p][i][j] = num
            '''
    
    return hist, hist_subplot

def insert_height_traits_into_betydb(in_dir, out_dir, str_date, param_percentile, sensor_d, convt):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    hist_e, hist_w = analysis_utils.load_histogram_from_both_npy(in_dir)
    
    out_file = os.path.join(out_dir, str_date+'_height.csv')
    csv = open(out_file, 'w')
    
    (fields, traits) = analysis_utils.get_traits_table_height()
        
    csv.write(','.join(map(str, fields)) + '\n')
        
    for j in range(0, PLOT_COL_NUM*PLOT_RANGE_NUM):
        targetHist_e = hist_e[j,:]
        targetHist_w = hist_w[j, :]
        plotNum = j+1
        if (targetHist_e.max() == 0) or (targetHist_w.max())==0:
            continue
        else:
            targetHist_e = targetHist_e/np.sum(targetHist_e)
            quantiles_e = np.cumsum(targetHist_e)
            b=np.arange(len(quantiles_e))
            c=b[quantiles_e>param_percentile]
            quantile_e = min(c)
            
            targetHist_w = targetHist_w/np.sum(targetHist_w)
            quantiles_w = np.cumsum(targetHist_w)
            b=np.arange(len(quantiles_w))
            c=b[quantiles_w>param_percentile]
            quantile_w = min(c)
            
            estHeight = (quantile_e + quantile_w)/2
                
            str_time = str_date+'T12:00:00'
            traits['local_datetime'] = str_time
            traits['canopy_height'] = str((B_F_SLOPE*float(estHeight) + B_F_OFFSET)/100.0)
            traits['site'] = analysis_utils.parse_site_from_plotNum_1728(plotNum, convt)
            trait_list = analysis_utils.generate_traits_list_height(traits)
            csv.write(','.join(map(str, trait_list)) + '\n')
    
    
    csv.close()
    #submitToBety(out_file)
    betydb.submit_traits(out_file, filetype='csv', betykey=betydb.get_bety_key(), betyurl=betydb.get_bety_url())
    
    return

def fail(reason):
    print >> sys.stderr, reason

if __name__ == "__main__":

    main()
