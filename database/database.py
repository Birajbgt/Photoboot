import sqlite3
import os



def create_table(table_name):
        # Create a table if it doesn't exist
    conn = sqlite3.connect('database/photo.db')

    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY,
            image_data BLOB NOT NULL
        );
    ''')
    conn.commit()
    
    
def upload_database(data, table_name):
    conn = sqlite3.connect('database/photo.db')

    cursor = conn.cursor()
    cursor.execute(f'''
            INSERT INTO {table_name} (image_data) VALUES (?);
        ''', (sqlite3.Binary(data),))
    conn.commit()
    


def download_images_from_database(table_name, output_folder):
    conn = sqlite3.connect('database/photo.db')
    cursor = conn.cursor()

    # Select all rows from the table
    cursor.execute(f'SELECT id, image_data FROM {table_name};')
    rows = cursor.fetchall()

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the rows and save images to the output folder
    for row in rows:
        image_id, image_data = row
        image_filename = os.path.join(output_folder, f'image_{image_id}.png')

        with open(image_filename, 'wb') as file:
            file.write(image_data)

    conn.close()
    #to download image
download_images_from_database("upload_table","photos_download")