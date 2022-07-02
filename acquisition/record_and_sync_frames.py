#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   record_and_sync_frames.py
@Time    :   2022/02/01 12:27:07
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com

Record frames with Basler (pypylon) with output trigger (audo abt)

'''
#%%
from pypylon import pylon
#from pypylon import genicam

import array
import sys
import errno
import os
import optparse
import hashlib
import serial
import cv2 #from scipy.misc import imsave
import threading
import time
from datetime import datetime
from libtiff import TIFF

from queue import Queue
import numpy as np
import multiprocessing as mp

#from samples.configurationeventprinter import ConfigurationEventPrinter
#from samples.imageeventprinter import ImageEventPrinter

current_ms_time = lambda: int(round(time.time() * 1000))


def extract_options(options):

    parser = optparse.OptionParser()
    parser.add_option('--save-dir', action="store", dest="save_dir", default="/home/julianarhee/Videos/basler", help="out path directory [default: /home/julianarhee/Videos/basler]")
    parser.add_option('--output-format', action="store", dest="output_format", type="choice", choices=['png', 'npz'], default='png', help="out file format, png or npz [default: png]")
    parser.add_option('--save', action='store_true', dest='save_images', default=False, help='Flag to save images to disk.')
    parser.add_option('--basename', action="store", dest="basename", default="acquisition", help="basename for saved acquisition")

    parser.add_option('--write-process', action="store_true", dest="save_in_separate_process", default=True, help="spawn process for disk-writer [default: True]")
    parser.add_option('--write-thread', action="store_false", dest="save_in_separate_process", help="spawn threads for disk-writer")
    parser.add_option('-f', '--frame-rate', action="store", dest="frame_rate", help="requested frame rate", type="float", default=60.0)

    parser.add_option('-W', '--width', action="store", dest="width", help="image width", type="int", default=960)
    parser.add_option('-H', '--height', action="store", dest="height", help="image height", type="int", default=960)
    parser.add_option('-x', '--exposure', action="store", dest="exposure", help="exposure (us))", type='float', default=16670)

    parser.add_option('-t', '--duration', action="store", dest="duration", help="recording duration (min)", type='float', default=np.inf)

    parser.add_option('--port', action="store", dest="port", help="port for arduino (default: /dev/ttyACM0)", default='/dev/ttyACM0')
    parser.add_option('--disable', action='store_false', dest='enable_framerate', default=True, help='Flag to disable acquisition frame rate setting.')

    parser.add_option('--no-trigger', action='store_false', dest='send_trigger', default=True, help='Flag to disable arduino trigger output.')

    (options, args) = parser.parse_args()

    return options


# ############################################
# Camera functions
# ############################################

def connect_to_camera(connect_retries=50, frame_rate=20., acquisition_line='Line3', enable_framerate=True,
                      width=1200, height=1200, exposure=16670):
    print('Searching for camera...')

    camera = None
    # get the camera list 
    print('Connecting to camera...')   
    n = 0
    while camera is None and n < connect_retries:
        try:
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            print(camera)
            #time.sleep(0.5)
            #camera.Open()
            #print("Bound to device:" % (camera.GetDeviceInfo().GetModelName()))

        except Exception as e:
            print('.')
            time.sleep(0.1)
            camera = None
            n += 1

    if camera is None:
        try:
            #import opencv_fallback

            #camera = opencv_fallback.Camera(0)

            print("Bound to OpenCV fallback camera.")
        except Exception as e2:
            print("Could not load OpenCV fallback camera")
            print(e2)
            exit()
    else:
        camera.Open()
        print("Bound to device: %s" % (camera.GetDeviceInfo().GetModelName()))

    camera.AcquisitionFrameRateEnable = enable_framerate
    camera.AcquisitionFrameRate = frame_rate
    if enable_framerate:
        camera.AcquisitionMode.SetValue('Continuous')
        print("Set acquisition frame rate: %.2f Hz" % camera.AcquisitionFrameRate())
        for trigger_type in ['FrameStart', 'FrameBurstStart']:
            camera.TriggerSelector = trigger_type
            camera.TriggerMode = "Off"
    else: 
        # Set  trigger
        camera.TriggerSelector = "FrameStart"
        camera.TriggerMode = "On"
    
    camera.TriggerSource.SetValue(acquisition_line)
    #camera.TriggerSelector.SetValue('AcquisitionStart')
    camera.TriggerActivation = 'RisingEdge'

    # Set IO lines:
    camera.LineSelector.SetValue(acquisition_line) # select GPIO 1
    camera.LineMode.SetValue('Input')     # Set as input
    #camera.LineStatus.SetValue(False)
    # Output:
    camera.LineSelector.SetValue('Line4')
    camera.LineMode.SetValue('Output')
    camera.LineSource.SetValue('UserOutput3') # Set source signal to User Output 1
    camera.UserOutputSelector.SetValue('UserOutput3')
    camera.UserOutputValue.SetValue(False)
       
    # Set image format:
    camera.Width.SetValue(width) #(960)
    camera.Height.SetValue(height) #(600)
    #camera.BinningHorizontalMode.SetValue('Sum')
    #camera.BinningHorizontal.SetValue(2)
    #camera.BinningVerticalMode.SetValue('Sum')
    #camera.BinningVertical.SetValue(2)
    camera.PixelFormat.SetValue('Mono8')

    camera.ExposureMode.SetValue('Timed')
    camera.ExposureTime.SetValue(exposure) #(40000)

    try:
        actual_framerate = camera.ResultingFrameRate.GetValue()
        assert camera.AcquisitionFrameRate() <= camera.ResultingFrameRate(), "Unable to acquieve desired frame rate (%.2f Hz)" % float(camera.AcquisitionFrameRate.GetValue())
    except AssertionError:
        camera.AcquisitionFrameRate.SetValue(float(camera.ResultingFrameRate.GetValue()))
        print("Set acquisition rate to: %.2f" % camera.AcquisitionFrameRate())
    print('Final frame rate: %.2f Hz' % (camera.AcquisitionFrameRate()))
     
    return camera

# compute a hash from the current time so that we don't accidentally overwrite old data
#run_hash = hashlib.md5(str(time.time())).hexdigest()

class SampleImageEventHandler(pylon.ImageEventHandler):
    def OnImageGrabbed(self, camera, grabResult):
        #print("CSampleImageEventHandler::OnImageGrabbed called.")
        camera.UserOutputValue.SetValue(True)
        #camera.UserOutputValue.SetValue(True)


# ############################################
# Main 
# ############################################
        
if __name__ == '__main__':

    optsE = extract_options(sys.argv[1:])

    # Arduino serial port
    port = optsE.port
    send_trigger=optsE.send_trigger
    
    # Camera settings:
    frame_rate = optsE.frame_rate
    frame_period = float(1/frame_rate)
    width = optsE.width
    height = optsE.height
    exposure = optsE.exposure
    duration = optsE.duration
    duration_sec = duration*60.0

    # Acquisition settings
    acquire_images = True
    save_images = optsE.save_images #True
    save_dir = optsE.save_dir
    output_format = optsE.output_format
    save_in_separate_process = optsE.save_in_separate_process   
    save_as_png = True #output_format=='tiff'
    basename = optsE.basename
    
    # Make the output path if it doesn't already exist
    tstamp_fmt = '%Y%m%d-%H%M%S'
    datestr = datetime.now().strftime(tstamp_fmt) 
    basename = '%s_%s' % (datestr, basename)
    dst_dir = os.path.join(save_dir, basename)
    frame_write_dir = os.path.join(dst_dir, 'frames')
    try:
        os.makedirs(frame_write_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
        pass
  
    frametimes_fpath = os.path.join(dst_dir, 'frame_times.txt')
    performance_fpath = os.path.join(dst_dir, 'performance.txt')

    # Set up stream      
    cv2.namedWindow('cam_window')
    r = np.random.rand(100,100)
    cv2.imshow('cam_window', r)

    time.sleep(1.0)

#%%
    # -------------------------------------------------------------
    # Set up serial connection
    # -------------------------------------------------------------
    if send_trigger:
        # port = "/dev/cu.usbmodem145201"
        baudrate = 115200
        print("# Please specify a port and a baudrate")
        print("# using hard coded defaults " + port + " " + str(baudrate))
        ser = serial.Serial(port, baudrate, timeout=0.5)
        time.sleep(1)
        #flushBuffer()
        sys.stdout.flush()
        print("Connected serial port...")
    else:
        ser=None


#%%
    # -------------------------------------------------------------
    # Camera Setup
    # ------------------------------------------------------------     
    enable_framerate = optsE.enable_framerate
    acquisition_line = 'Line3'
    camera = None
    if acquire_images:
        camera = connect_to_camera(frame_rate=frame_rate, acquisition_line=acquisition_line, 
                                   enable_framerate=enable_framerate, width=width, height=height,
                            exposure=exposure) 
    # Attach event handlers:
    camera.RegisterImageEventHandler(SampleImageEventHandler(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)

    time.sleep(1)
    print("Camera ready!")

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) #GrabStrategy_OneByOne)
    # converting to opencv bgr format  
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

#%%
    # -------------------------------------------------------------
    # Set up a thread to write stuff to disk
    # -------------------------------------------------------------
    if save_in_separate_process:
        im_queue = mp.Queue()
    else:
        im_queue = Queue()

    def save_images_to_disk():
        print('Disk-saving thread active...')

        # Create frame metadata file:
        date_fmt = '%Y%m%d_%H%M%S%f'
        tstamp = datetime.now().strftime(date_fmt)
        
        #serial_outfile = os.path.join(dst_dir, '%s_frame_metadata_%s.txt' % (basename, tstamp))
        frametimes_fpath = os.path.join(dst_dir, 'frame_times.txt')
        print("Created outfile: %s" % frametimes_fpath) #serial_outfile)
        serial_file = open(frametimes_fpath, 'w+') #open(serial_outfile, 'w+')
        serial_file.write('count\tframe\tframe_ID\tframe_tstamp\timage_num\tacq_trigger\tframe_trigger\trelative_time\trelative_camera_time\n')

        n = 0
        result = im_queue.get()
        while result is not None: 
            (im_array, metadata) = result
            if n==0:
                start_time = time.perf_counter() #process_time() #clock() 
                cam_start_time = metadata['tstamp']

            #name = '%i_%i_%i' % (n, metadata['ID'], metadata['tstamp'])
            name = 'frame-%06d' % n
            if save_as_png:
                fpath = os.path.join(frame_write_dir, '%s.png' % name)
                #tiff = TIFF.open(fpath, mode='w')
                #tiff.write_image(im_array)
                #tiff.close()
                cv2.imwrite(fpath, im_array)
            else:
                fpath = os.path.join(frame_write_dir, '%s.npz' % name)
                np.savez_compressed(fpath, im_array)

            serial_file.write('\t'.join([str(s) for s in [metadata['frame_count'], n, metadata['ID'], metadata['tstamp'], metadata['number'], metadata['acq_trigger'], metadata['frame_trigger'], str(time.process_time()-start_time), (metadata['tstamp']-cam_start_time)/1E9]]) + '\n')
            n += 1
            result = im_queue.get()

        disk_writer_alive = False 
        print('Disk-saving thread inactive...')
        serial_file.flush()
        serial_file.close()
        print("Closed data file...")

    if save_in_separate_process:
        disk_writer = mp.Process(target=save_images_to_disk)
    else:
        disk_writer = threading.Thread(target=save_images_to_disk)

    if save_images:
        disk_writer.daemon = True
        disk_writer.start()

#%%
    # Frame triggering  
    camera.LineSelector.SetValue(acquisition_line) 
    sync_line = camera.LineSelector.GetValue()
    sync_state = camera.LineStatus.GetValue()
    print("Waiting for Acquisition Start trigger...", sync_state)
    while sync_state is False: 
        #print("[%s] trigger" % sync_line, sync_state)
        sync_state = camera.LineStatus.GetValue()
    print("... ... MW trigger received!")
    camera.AcquisitionStart.Execute()

    #while True:
        #camera.WaitForFrameTriggerReady(100)

    # set up timers and counters
    nframes = 0
    t = 0
    last_t = None
    report_interval=1.0
    report_period = frame_rate*report_interval # frames
    timeout_time = 1000
    start_time = time.perf_counter() # start timer in frac. seconds

    if send_trigger:
        #byte_string = str.encode('S')
        ser.write(str.encode('S')) #('S')#start arduino trigger
        print('Triggered arduino....')

    # -------------------------------------------------------------
    # Start acquiring
    # -------------------------------------------------------------
    print('Beginning imaging [Hit ESC to quit]...')
    while camera.IsGrabbing():
        t = time.time() 
        #while camera.IsGrabbing():
        # Grab a frame:
        #camera.WaitForFrameTriggerReady(100)
        res = camera.RetrieveResult(timeout_time, pylon.TimeoutHandling_ThrowException)
        if res.GrabSucceeded():
            # Access img data:
            im_native = res.Array
            im_to_show = converter.Convert(res)
            im_array = im_to_show.GetArray()
            frame_state = camera.UserOutputValue.GetValue()
            meta = {'tstamp': res.TimeStamp, 
                    'ID': res.ID,
                    'number': res.ImageNumber,
                    'frame_count': nframes,
                    'acq_trigger': sync_state,
                    'frame_trigger': frame_state}
            if save_images:
                im_queue.put((im_native, meta))
            nframes += 1

        # Show image:
        cv2.imshow('cam_window', im_array)
        camera.UserOutputValue.SetValue(False)

        # Break out of the while loop if ESC registered
        elapsed_time = time.perf_counter() - start_time
        key = cv2.waitKey(1)
        sync_state = camera.LineStatus.GetValue()
        if key == 27 or sync_state is False or (elapsed_time>duration_sec): # ESC
            break
        res.Release()

        if nframes % report_period == 0:
            if last_t is not None:
                print('avg frame rate: %f [Hit ESC to quit]' % (report_period / (t - last_t)))
                print('ID: %i, nframes: %i, %s' % (meta['ID'], nframes, meta['tstamp']) )
            last_t = t

    camera.AcquisitionStop.Execute()
    #camera.AcquisitionStart.Execute()

    # Relase the resource:
    camera.UserOutputValue.SetValue(False) 
    camera.StopGrabbing()
    cv2.destroyAllWindows()
    camera.Close() 

    if im_queue is not None:
        im_queue.put(None)


    print(str.format('total recording time: {} min',(elapsed_time/60)))

#%%
    # Stop arduino
    if send_trigger:
        ser.write(str.encode('F')) #('S')#start arduino trigger
        #ser.write('F')#start arduino trigger
        print('Stopped arduino....')

    print('Acquisition Finished!')
    #output performance
    acq_duration=time.perf_counter()-start_time
    print('Total Time: %.3f sec' % acq_duration)
    expected_frames=int(np.floor(np.around(acq_duration,2)/frame_period))
    print('Actual Frame Count = '+str(nframes+1))
    print('Expected Frame Count = '+str(expected_frames))

    # write performance to file
    performance_file = open(performance_fpath,'w+')
    performance_file.write('frame_rate\tframe_period\tacq_duration\tframe_count\texpected_frame_count\tmissing_frames\n')
    performance_file.write('%10.4f\t%10.4f\t%10.4f\t%i\t%i\t%i\n'%\
        (frame_rate, frame_period, acq_duration, nframes, expected_frames, expected_frames-nframes))
    performance_file.close()

    # Save images in buffer
    print("Saving images, recording dur was: %.2f min" % (elapsed_time/60.)) 
    if save_images:
        hang_time = time.time()
        nag_time = 0.05

        sys.stdout.write('Waiting for disk writer to catch up (this may take a while)...')
        sys.stdout.flush()
        waits = 0
        while not im_queue.empty():
            now = time.time()
            if (now - hang_time) > nag_time:
                sys.stdout.write('.')
                sys.stdout.flush()
                hang_time = now
                waits += 1
        print(waits)
        print("\n")

        if not im_queue.empty():
            print("WARNING: not all images have been saved to disk!")

        disk_writer_alive = False
        if save_in_separate_process and disk_writer is not None:
            print("Terminating disk writer...")
            disk_writer.join()
            # disk_writer.terminate() 
        # disk_writer.join()
        print('Disk writer terminated')        


    # Stop arduino
    if send_trigger:
        ser.write(str.encode('F')) #('S')#start arduino trigger
        #ser.write('F'.encode())
        #serial_file.flush()
        #serial_file.close()
        print('Stopped arduino....')



