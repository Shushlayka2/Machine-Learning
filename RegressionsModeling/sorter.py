import csv
import operator

class Sorter:
    
    def sort_csv_file(self, file_prev_name, file_after_name):
        f = open(file_prev_name, 'r')
        data = csv.reader(f)
        sorted_data = sorted(data, key=operator.itemgetter(1))
        f.close()
        f = open(file_after_name, 'w', newline='')
        writer = csv.writer(f)
        writer.writerows(sorted_data)