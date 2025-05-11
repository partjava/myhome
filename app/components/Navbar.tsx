'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import CategoryDisplay from './CategoryDisplay';

export default function Navbar() {
  const pathname = usePathname();
  const [showCategories, setShowCategories] = useState(false);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (e.clientY < 60) {
        setVisible(true);
      } else {
        setVisible(false);
      }
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <nav
      className="bg-white shadow-lg fixed w-full top-0 z-50"
      style={{
        transition: 'transform 0.3s',
        transform: visible ? 'translateY(0)' : 'translateY(-100%)',
      }}
    >
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            {/* Logo */}
            <div className="flex items-center h-16 border-b-2 border-transparent">
              <Link href="/" className="text-xl font-medium text-indigo-600">
                学习笔记
              </Link>
            </div>

            {/* 主导航 */}
            <div className="ml-8 flex space-x-8 h-full">
              <Link
                href="/"
                className={`inline-flex items-center px-1 h-16 border-b-2 text-sm font-medium ${
                  pathname === '/'
                    ? 'border-indigo-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                }`}
              >
                首页
              </Link>
              
              <div 
                className="relative h-full flex items-center"
                onMouseEnter={() => setShowCategories(true)}
                onMouseLeave={() => setShowCategories(false)}
              >
                <Link
                  href="/study"
                  className={`inline-flex items-center px-1 h-16 border-b-2 text-sm font-medium ${
                    pathname.startsWith('/study')
                      ? 'border-indigo-500 text-gray-900'
                      : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                  }`}
                >
                  学习
                </Link>
                
                {showCategories && <CategoryDisplay />}
              </div>

              <Link
                href="/notes"
                className={`inline-flex items-center px-1 h-16 border-b-2 text-sm font-medium ${
                  pathname.startsWith('/notes')
                    ? 'border-indigo-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                }`}
              >
                笔记
              </Link>

              <Link
                href="/profile"
                className={`inline-flex items-center px-1 h-16 border-b-2 text-sm font-medium ${
                  pathname === '/profile'
                    ? 'border-indigo-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                }`}
              >
                个人主页
              </Link>
            </div>
          </div>

          {/* 用户菜单 */}
          <div className="flex items-center">
            <Link
              href="/login"
              className="text-sm font-medium text-gray-500 hover:text-gray-700 mr-4"
            >
              登录
            </Link>
            <Link
              href="/register"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700"
            >
              注册
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
} 