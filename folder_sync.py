import os
import shutil
import argparse
import schedule
import time
import logging
import datetime

def synchronize_folders(source_folder, replica_folder, log_file):
    # Create log file
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    # print.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    def sync():
        # Remove all files and folders from replica folder
        shutil.rmtree(replica_folder)
        os.makedirs(replica_folder)

        # Copy all files and folders from source folder to replica folder
        for root, dirs, files in os.walk(source_folder):
            now = datetime.datetime.now()
            print("Schedule to syncronize the folders: ", now)
            for file in files:
                source_path = os.path.join(root, file)
                replica_path = os.path.join(replica_folder, os.path.relpath(source_path, source_folder))
                shutil.copy2(source_path, replica_path)
                logging.info(f'Copied: {source_path} -> {replica_path}')
                print(f'Copied: {source_path} -> {replica_path}')
                

    # Perform initial synchronization
    sync()
    logging.info('Initial synchronization complete')
    print('Initial synchronization complete')

    # Schedule periodic synchronization
    schedule.every(10).seconds.do(sync)
   
   
    # Keep the script running to allow periodic synchronization
    while True:
        schedule.run_pending()
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description='Folder Synchronization')
    parser.add_argument('source', help='./source')
    parser.add_argument('replica', help='./replica')
    parser.add_argument('log_file', help='./log_file.log')

    args = parser.parse_args()

    synchronize_folders(args.source, args.replica, args.log_file)


if __name__ == '__main__':
    main()


# write this command to run the project in the terminal

# python3 folder_sync.py ./source ./replica ./log_file.log

