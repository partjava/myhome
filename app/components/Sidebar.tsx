'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Button } from 'antd';
import { navigationItems, NavigationItems, NavigationItem } from '../data/navigation';

export default function Sidebar() {
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null);
  const [expandedItem, setExpandedItem] = useState<string | null>(null);
  const pathname = usePathname();
  const initialized = useRef(false);
  const itemRefs = useRef<{ [key: string]: HTMLButtonElement | null }>({});

  const toggleCategory = (category: string) => {
    if (expandedCategory === category) {
      setExpandedCategory(null);
    } else {
      setExpandedCategory(category);
    }
  };

  const toggleItem = (itemName: string) => {
    if (expandedItem === itemName) {
      setExpandedItem(null);
    } else {
      setExpandedItem(itemName);
    }
  };

  useEffect(() => {
    if (!initialized.current && typeof window !== 'undefined') {
      setExpandedCategory(localStorage.getItem('sidebar_expandedCategory'));
      setExpandedItem(localStorage.getItem('sidebar_expandedItem'));
      initialized.current = true;
    }
  }, []);

  useEffect(() => {
    if (initialized.current) {
      if (expandedCategory !== null) {
        localStorage.setItem('sidebar_expandedCategory', expandedCategory);
      } else {
        localStorage.removeItem('sidebar_expandedCategory');
      }
      if (expandedItem !== null) {
        localStorage.setItem('sidebar_expandedItem', expandedItem);
      } else {
        localStorage.removeItem('sidebar_expandedItem');
      }
    }
  }, [expandedCategory, expandedItem]);

  // 自动滚动到高亮项
  useEffect(() => {
    const ref = Object.values(itemRefs.current).find(
      (el) => el && el.classList.contains('text-blue-600')
    );
    if (ref) {
      ref.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  }, [pathname]);

  return (
    <div className="w-56 bg-white shadow-lg fixed h-full left-0 overflow-y-auto">
      <div className="px-4 py-5">
        <h2 className="text-lg font-medium text-gray-900">分类导航</h2>
      </div>
      <nav className="mt-5 px-2">
        {Object.entries(navigationItems).map(([category, items]: [string, NavigationItem[]]) => (
          <div key={category} className="mb-4">
            <button
              onClick={() => toggleCategory(category)}
              className="w-full flex items-center justify-between px-2 py-2 text-sm font-medium text-gray-600 rounded-md hover:bg-gray-50 hover:text-gray-900"
            >
              <span>{category}</span>
              <span className="transform transition-transform duration-200">
                {expandedCategory === category ? '▼' : '▶'}
              </span>
            </button>
            {expandedCategory === category && (
              <div className="ml-4 mt-2">
                {items.map((item: NavigationItem) => (
                  <div key={item.code}>
                    <button
                      onClick={() => toggleItem(item.name)}
                      className="w-full flex items-center justify-between px-2 py-1 text-sm text-gray-700 hover:bg-gray-100 rounded"
                      ref={(el: HTMLButtonElement | null) => {
                        if (el) {
                          itemRefs.current[item.name] = el;
                        }
                      }}
                    >
                      <span>{item.name}</span>
                      {item.subitems && (
                        <span>{expandedItem === item.name ? '▼' : '▶'}</span>
                      )}
                    </button>
                    {item.subitems && expandedItem === item.name && (
                      <div className="ml-4 mt-1">
                        {item.subitems.map((sub: { name: string; href: string }) => (
                          <Link
                            key={sub.name}
                            href={sub.href}
                            className={`block px-2 py-1 text-sm rounded hover:bg-blue-50 ${pathname === sub.href ? 'text-blue-600 font-bold' : 'text-gray-600'}`}
                          >
                            {sub.name}
                          </Link>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </nav>
    </div>
  );
} 