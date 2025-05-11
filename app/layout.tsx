import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import 'antd/dist/reset.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: '学习笔记分享平台',
  description: '一个用于学习和分享技术知识的平台',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh">
      <body className={inter.className}>
        <Navbar />
        <div className="flex">
          <Sidebar />
          <main className="flex-1 ml-64 min-h-screen bg-gray-50">
        {children}
          </main>
        </div>
      </body>
    </html>
  )
}
