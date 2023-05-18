import os
import pickle
import argparse
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.file']
# //AIzaSyACbrrWDeFbo7Ta9WD3cZoGSMIHl9ruIXQ

def authenticate():
    creds = None

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return creds


def create_file(service, filename, filepath, folder_id=None):
    file_metadata = {'name': filename, 'parents': [folder_id] if folder_id else []}
    media = MediaFileUpload(filepath, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'Successfully created file with ID: {file.get("id")}')


def retrieve_files(service):
    results = service.files().list().execute()
    files = results.get('files', [])
    if not files:
        print('No files found.')
    else:
        print('Files:')
        for file in files:
            print(f'{file["name"]} ({file["id"]})')


def update_file(service, file_id, new_filename):
    file_metadata = {'name': new_filename}
    file = service.files().update(fileId=file_id, body=file_metadata).execute()
    print(f'Successfully updated file: {file["name"]}')


def delete_file(service, file_id):
    service.files().delete(fileId=file_id).execute()
    print('File deleted successfully')


def main():
    parser = argparse.ArgumentParser(description='Google Drive API Example')
    parser.add_argument('action', choices=['create', 'retrieve', 'update', 'delete'],
                        help='Specify the action to perform')
    parser.add_argument('--filename', help='Specify the name of the file')
    parser.add_argument('--filepath', help='Specify the path of the file to upload')
    parser.add_argument('--fileid', help='Specify the ID of the file to update or delete')
    parser.add_argument('--newfilename', help='Specify the new name for the file')

    args = parser.parse_args()

    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    if args.action == 'create':
        create_file(service, args.filename, args.filepath)
    elif args.action == 'retrieve':
        retrieve_files(service)
    elif args.action == 'update':
        update_file(service, args.fileid, args.newfilename)
    elif args.action == 'delete':
        delete_file(service, args.fileid)


if __name__ == '__main__':
    main()
