'use client';
import Link from 'next/link';
import { navigationItems, NavigationItem } from '../data/navigation';

export default function CategoryDisplay() {
  return (
    <div className="fixed left-0 right-0 top-16 bg-white shadow-lg z-50">
      <div className="max-w-[1400px] mx-auto py-6">
        <div className="grid grid-cols-4 gap-8">
          {Object.entries(navigationItems).map(([category, items]: [string, NavigationItem[]]) => (
            <div key={category} className="col-span-1">
              <h3 className="text-base font-medium text-gray-900 mb-4 pb-2 border-b">
                {category}
              </h3>
              <div className="grid grid-cols-1 gap-y-4">
                {items.map((item: NavigationItem) => (
                  <Link
                    key={item.code}
                    href={item.subitems ? `/study/cpp/${item.code}` : `/study/${category}/${item.code}`}
                    className="group flex items-center space-x-3 text-gray-600 hover:text-blue-600"
                  >
                    <span className="flex-shrink-0 w-6 h-6 flex items-center justify-center bg-gray-100 text-sm text-gray-500 group-hover:bg-blue-50 group-hover:text-blue-600 rounded">
                      {item.code}
                    </span>
                    <span className="text-sm">{item.name}</span>
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 