'use client';
import { useState } from 'react';
import NoteEditor from '../components/NoteEditor';

interface Note {
  id: string;
  title: string;
  content: string;
  createdAt: Date;
}

export default function NotesPage() {
  const [notes, setNotes] = useState<Note[]>([]);

  const handleSaveNote = (title: string, content: string) => {
    const newNote: Note = {
      id: Date.now().toString(),
      title,
      content,
      createdAt: new Date(),
    };
    setNotes([newNote, ...notes]);
  };

  return (
    <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <div className="px-4 py-6 sm:px-0">
        <div className="border-4 border-dashed border-gray-200 rounded-lg p-4">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">我的笔记</h1>
          
          {/* 笔记编辑器 */}
          <div className="mb-8">
            <NoteEditor onSave={handleSaveNote} />
          </div>

          {/* 笔记列表 */}
          <div className="space-y-4">
            {notes.map((note) => (
              <div
                key={note.id}
                className="bg-white overflow-hidden shadow rounded-lg"
              >
                <div className="px-4 py-5 sm:p-6">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">
                    {note.title}
                  </h3>
                  <div className="mt-2 max-w-xl text-sm text-gray-500">
                    <p>{note.content}</p>
                  </div>
                  <div className="mt-3 text-sm text-gray-500">
                    {note.createdAt.toLocaleString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
} 