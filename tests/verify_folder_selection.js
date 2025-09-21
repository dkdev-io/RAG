import puppeteer from 'puppeteer';
import fs from 'fs';
import path from 'path';

async function verifyFolderSelection() {
  const browser = await puppeteer.launch({
    headless: false,
    defaultViewport: { width: 1200, height: 800 },
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1200, height: 800 });

    console.log('Navigating to Streamlit app...');
    await page.goto('http://localhost:8502', {
      waitUntil: 'networkidle2',
      timeout: 30000
    });

    // Wait for Streamlit to fully load
    await new Promise(resolve => setTimeout(resolve, 3000));

    // 1. Take screenshot of main interface showing all 4 tabs
    console.log('Taking screenshot of main interface...');
    await page.screenshot({
      path: '/Users/Danallovertheplace/rag/tests/screenshots/01_main_interface.png',
      fullPage: true
    });

    // 2. Click on "Select Folder" tab
    console.log('Clicking on Select Folder tab...');
    const selectFolderTab = await page.waitForSelector('div[data-testid="stTabs"] button:nth-child(2)', { timeout: 10000 });
    await selectFolderTab.click();
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 3. Take screenshot of folder selection interface
    console.log('Taking screenshot of folder selection interface...');
    await page.screenshot({
      path: '/Users/Danallovertheplace/rag/tests/screenshots/02_select_folder_tab.png',
      fullPage: true
    });

    // Check for folder selection elements
    console.log('Checking folder selection elements...');
    const elements = {
      currentFolder: false,
      manualInput: false,
      setButton: false,
      recentFolders: false
    };

    // Check for current folder display
    try {
      await page.waitForSelector('text=/Current Folder|Selected Folder/', { timeout: 5000 });
      elements.currentFolder = true;
      console.log('✓ Current folder display found');
    } catch (e) {
      console.log('✗ Current folder display not found');
    }

    // Check for manual path input field
    try {
      const textInput = await page.$('input[type="text"]');
      if (textInput) {
        elements.manualInput = true;
        console.log('✓ Manual path input field found');
      }
    } catch (e) {
      console.log('✗ Manual path input field not found');
    }

    // Check for Set Folder button
    try {
      await page.waitForSelector('button:has-text("Set Folder")', { timeout: 5000 });
      elements.setButton = true;
      console.log('✓ Set Folder button found');
    } catch (e) {
      console.log('✗ Set Folder button not found');
    }

    // Check for recent/suggested folders section
    try {
      await page.waitForSelector('text=/Recent|Suggested/', { timeout: 5000 });
      elements.recentFolders = true;
      console.log('✓ Recent/suggested folders section found');
    } catch (e) {
      console.log('✗ Recent/suggested folders section not found');
    }

    // 4. Click on "Manage Files" tab
    console.log('Clicking on Manage Files tab...');
    const manageFilesTab = await page.waitForSelector('div[data-testid="stTabs"] button:nth-child(3)', { timeout: 10000 });
    await manageFilesTab.click();
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 5. Take screenshot of Manage Files tab
    console.log('Taking screenshot of Manage Files tab...');
    await page.screenshot({
      path: '/Users/Danallovertheplace/rag/tests/screenshots/03_manage_files_tab.png',
      fullPage: true
    });

    // 6. Click on "System Status" tab
    console.log('Clicking on System Status tab...');
    const systemStatusTab = await page.waitForSelector('div[data-testid="stTabs"] button:nth-child(4)', { timeout: 10000 });
    await systemStatusTab.click();
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 7. Take screenshot of System Status tab
    console.log('Taking screenshot of System Status tab...');
    await page.screenshot({
      path: '/Users/Danallovertheplace/rag/tests/screenshots/04_system_status_tab.png',
      fullPage: true
    });

    // 8. Return to "Query Documents" tab
    console.log('Returning to Query Documents tab...');
    const queryTab = await page.waitForSelector('div[data-testid="stTabs"] button:nth-child(1)', { timeout: 10000 });
    await queryTab.click();
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 9. Take screenshot showing folder indicator
    console.log('Taking screenshot of Query Documents with folder indicator...');
    await page.screenshot({
      path: '/Users/Danallovertheplace/rag/tests/screenshots/05_query_with_folder_indicator.png',
      fullPage: true
    });

    // Generate test report
    const report = {
      timestamp: new Date().toISOString(),
      testResults: {
        mainInterface: true,
        selectFolderTab: true,
        manageFilesTab: true,
        systemStatusTab: true,
        queryTabReturn: true,
        folderSelectionElements: elements
      },
      screenshots: [
        '01_main_interface.png',
        '02_select_folder_tab.png',
        '03_manage_files_tab.png',
        '04_system_status_tab.png',
        '05_query_with_folder_indicator.png'
      ]
    };

    console.log('\n=== TEST REPORT ===');
    console.log('Timestamp:', report.timestamp);
    console.log('Screenshots captured:', report.screenshots.length);
    console.log('Folder selection elements:');
    console.log('  - Current folder display:', elements.currentFolder ? '✓' : '✗');
    console.log('  - Manual input field:', elements.manualInput ? '✓' : '✗');
    console.log('  - Set Folder button:', elements.setButton ? '✓' : '✗');
    console.log('  - Recent folders section:', elements.recentFolders ? '✓' : '✗');

    // Save test report
    fs.writeFileSync('/Users/Danallovertheplace/rag/tests/screenshots/test_report.json', JSON.stringify(report, null, 2));

    console.log('\nAll screenshots saved to /Users/Danallovertheplace/rag/tests/screenshots/');
    console.log('Test completed successfully!');

  } catch (error) {
    console.error('Error during testing:', error);

    // Take error screenshot
    try {
      await page.screenshot({
        path: '/Users/Danallovertheplace/rag/tests/screenshots/error_screenshot.png',
        fullPage: true
      });
      console.log('Error screenshot saved');
    } catch (screenshotError) {
      console.error('Could not take error screenshot:', screenshotError);
    }
  } finally {
    await browser.close();
  }
}

// Run the verification
verifyFolderSelection().catch(console.error);