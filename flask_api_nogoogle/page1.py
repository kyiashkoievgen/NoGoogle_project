# page1.py
import sqlite3

from flask import Flask, request
from flask import render_template

from scan import scan_input
from search import search_book, search_page, txt_to_html

# from app import app

app = Flask(__name__)


@app.route('/nogoogle.com')
def welcome_text():
    return render_template("index.html")


@app.route('/nogoogle.com/book', methods=['POST'])
def s_book():
    txt = request.values['txt']
    book_id, book_list = search_book(txt)
    book_list_link = '<table>'
    for i in range(len(book_list)):
        book_list_link += f'<tr><td><A href="/nogoogle.com/search_txt?book_id={book_id[i]}&txt={txt}"  target="text">' \
                          f'{book_list[i]}"</a><td><tr>'
    book_list_link += '</table>'
    return book_list_link


@app.route('/nogoogle.com/search_txt', methods=['GET'])
def s_text():
    txt = request.values['txt']
    book_id = request.values['book_id']
    file_name, pages = search_page(book_id, txt)
    book_str = ''
    for page in pages:
        book_str += f'<a href=#{page}>{page}, </a>'
    return render_template("book_frame.html", book_txt=txt_to_html(file_name, pages), book_str=book_str)


if __name__ == "__main__":
    app.run(debug=True)
